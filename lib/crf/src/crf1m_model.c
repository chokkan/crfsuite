#include "os.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cqdb.h>

#include "crf.h"
#include "crf1m.h"

#define	FILEMAGIC		"lCRF"
#define	MODELTYPE		"FOMC"
#define	VERSION_NUMBER	(100)
#define	CHUNK_LABELREF	"LFRF"
#define	CHUNK_ATTRREF	"AFRF"
#define	CHUNK_FEATURE	"FEAT"

enum {
	WSTATE_NONE,
	WSTATE_LABELS,
	WSTATE_ATTRS,
	WSTATE_LABELREFS,
	WSTATE_ATTRREFS,
	WSTATE_FEATURES,
};

typedef struct {
	uint8_t		magic[4];			/* File magic. */
	uint32_t	size;				/* File size. */
	uint8_t		type[4];			/* Model type */
	uint32_t	version;			/* Version number. */
	uint32_t	num_features;		/* Number of features. */
	uint32_t	num_labels;			/* Number of labels. */
	uint32_t	num_attrs;			/* Number of attributes. */
	uint32_t	off_features;		/* Offset to features. */
	uint32_t	off_labels;			/* Offset to label CQDB. */
	uint32_t	off_attrs;			/* Offset to attribute CQDB. */
	uint32_t	off_labelrefs;		/* Offset to label feature references. */
	uint32_t	off_attrrefs;		/* Offset to attribute feature references. */
} header_t;

typedef struct {
	uint8_t		chunk[4];			/* Chunk id */
	uint32_t	size;				/* Chunk size. */
	uint32_t	num;				/* Number of items. */
	uint32_t	offsets[1];			/* Offsets. */
} featureref_header_t;

typedef struct {
	uint8_t		chunk[4];			/* Chunk id */
	uint32_t	size;				/* Chunk size. */
	uint32_t	num;				/* Number of items. */
} feature_header_t;

struct tag_crf1mm {
	uint8_t*	buffer;
	uint32_t	size;
	header_t*	header;
	cqdb_t*		labels;
	cqdb_t*		attrs;
};

struct tag_crf1mmw {
	FILE *fp;
	int state;
	header_t header;
	cqdb_writer_t* dbw;
	featureref_header_t* href;
	feature_header_t* hfeat;
};


enum {
	KT_GLOBAL = 'A',
	KT_NUMATTRS,
	KT_NUMLABELS,
	KT_STR2LID,
	KT_LID2STR,
	KT_STR2AID,
	KT_FEATURE,
};

static int write_uint8(FILE *fp, uint8_t value)
{
	return fwrite(&value, sizeof(value), 1, fp) == 1 ? 0 : 1;
}

static int read_uint8(uint8_t* buffer, uint8_t* value)
{
	*value = *buffer;
	return sizeof(*value);
}

static int write_uint32(FILE *fp, uint32_t value)
{
	return fwrite(&value, sizeof(value), 1, fp) == 1 ? 0 : 1;
}

static int read_uint32(uint8_t* buffer, uint32_t* value)
{
	*value = *(uint32_t*)buffer;
	return sizeof(*value);
}

static int write_uint8_array(FILE *fp, uint8_t *array, size_t n)
{
	size_t i;
	int ret = 0;
	for (i = 0;i < n;++i) {
		ret |= write_uint8(fp, array[i]);
	}
	return ret;
}

static int read_uint8_array(uint8_t* buffer, uint8_t *array, size_t n)
{
	size_t i;
	int ret = 0;
	for (i = 0;i < n;++i) {
		int size = read_uint8(buffer, &array[i]);
		buffer += size;
		ret += size;
	}
	return ret;
}

static void write_float(FILE *fp, floatval_t value)
{
	fwrite(&value, sizeof(value), 1, fp);
}

static int read_float(uint8_t* buffer, floatval_t* value)
{
	*value = *(floatval_t*)buffer;
	return sizeof(*value);
}

crf1mmw_t* crf1mmw(const char *filename)
{
	header_t *header = NULL;
	crf1mmw_t *writer = NULL;

	/* Create a writer instance. */
	writer = (crf1mmw_t*)calloc(1, sizeof(crf1mmw_t));
	if (writer == NULL) {
		goto error_exit;
	}

	/* Open the file for writing. */
	writer->fp = fopen(filename, "wb");
	if (writer->fp == NULL) {
		goto error_exit;
	}

	/* Fill the members in the header. */
	header = &writer->header;
	strncpy(header->magic, FILEMAGIC, 4);
	strncpy(header->type, MODELTYPE, 4);
	header->version = VERSION_NUMBER;

	/* Advance the file position to skip the file header. */
	if (fseek(writer->fp, sizeof(header_t), SEEK_CUR) != 0) {
		goto error_exit;
	}

	return writer;

error_exit:
	if (writer != NULL) {
		if (writer->fp != NULL) {
			fclose(writer->fp);
		}
		free(writer);
	}
	return NULL;
}

int crf1mmw_close(crf1mmw_t* writer)
{
	FILE *fp = writer->fp;
	header_t *header = &writer->header;

	/* Store the file size. */
	header->size = (uint32_t)ftell(fp);

	/* Move the file position to the head. */
	if (fseek(fp, 0, SEEK_SET) != 0) {
		goto error_exit;
	}

	/* Write the file header. */
	write_uint8_array(fp, header->magic, sizeof(header->magic));
	write_uint32(fp, header->size);
	write_uint8_array(fp, header->type, sizeof(header->type));
	write_uint32(fp, header->version);
	write_uint32(fp, header->num_features);
	write_uint32(fp, header->num_labels);
	write_uint32(fp, header->num_attrs);
	write_uint32(fp, header->off_features);
	write_uint32(fp, header->off_labels);
	write_uint32(fp, header->off_attrs);
	write_uint32(fp, header->off_labelrefs);
	write_uint32(fp, header->off_attrrefs);

	/* Check for any error occurrence. */
	if (ferror(fp)) {
		goto error_exit;
	}

	/* Close the writer. */
	fclose(fp);
	free(writer);
	return 0;

error_exit:
	if (writer != NULL) {
		if (writer->fp != NULL) {
			fclose(writer->fp);
		}
		free(writer);
	}
	return 1;
}

int crf1mmw_open_labels(crf1mmw_t* writer, int num_labels)
{
	/* Check if we aren't writing anything at this moment. */
	if (writer->state != WSTATE_NONE) {
		return 1;
	}

	/* Store the current offset. */
	writer->header.off_labels = (uint32_t)ftell(writer->fp);

	/* Open a CQDB chunk for writing. */
	writer->dbw = cqdb_writer(writer->fp, 0);
	if (writer->dbw == NULL) {
		writer->header.off_labels = 0;
		return 1;
	}

	writer->state = WSTATE_LABELS;
	writer->header.num_labels = num_labels;
	return 0;
}

int crf1mmw_close_labels(crf1mmw_t* writer)
{
	/* Make sure that we are writing labels. */
	if (writer->state != WSTATE_LABELS) {
		return 1;
	}

	/* Close the CQDB chunk. */
	if (cqdb_writer_close(writer->dbw)) {
		return 1;
	}

	writer->dbw = NULL;
	writer->state = WSTATE_NONE;
	return 0;
}

int crf1mmw_put_label(crf1mmw_t* writer, int lid, const char *value)
{
	/* Make sure that we are writing labels. */
	if (writer->state != WSTATE_LABELS) {
		return 1;
	}

	/* Put the label. */
	if (cqdb_writer_put(writer->dbw, value, lid)) {
		return 1;
	}

	return 0;
}

int crf1mmw_open_attrs(crf1mmw_t* writer, int num_attrs)
{
	/* Check if we aren't writing anything at this moment. */
	if (writer->state != WSTATE_NONE) {
		return 1;
	}

	/* Store the current offset. */
	writer->header.off_attrs = (uint32_t)ftell(writer->fp);

	/* Open a CQDB chunk for writing. */
	writer->dbw = cqdb_writer(writer->fp, 0);
	if (writer->dbw == NULL) {
		writer->header.off_attrs = 0;
		return 1;
	}

	writer->state = WSTATE_ATTRS;
	writer->header.num_attrs = num_attrs;
	return 0;
}

int crf1mmw_close_attrs(crf1mmw_t* writer)
{
	/* Make sure that we are writing attributes. */
	if (writer->state != WSTATE_ATTRS) {
		return 1;
	}

	/* Close the CQDB chunk. */
	if (cqdb_writer_close(writer->dbw)) {
		return 1;
	}

	writer->dbw = NULL;
	writer->state = WSTATE_NONE;
	return 0;
}

int crf1mmw_put_attr(crf1mmw_t* writer, int aid, const char *value)
{
	/* Make sure that we are writing labels. */
	if (writer->state != WSTATE_ATTRS) {
		return 1;
	}

	/* Put the attribute. */
	if (cqdb_writer_put(writer->dbw, value, aid)) {
		return 1;
	}

	return 0;
}

int crf1mmw_open_labelrefs(crf1mmw_t* writer, int num_labels)
{
	FILE *fp = writer->fp;
	featureref_header_t* href = NULL;
	size_t size = sizeof(featureref_header_t) + sizeof(uint32_t) * (num_labels-1);

	/* Check if we aren't writing anything at this moment. */
	if (writer->state != WSTATE_NONE) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Allocate a feature reference array. */
	href = (featureref_header_t*)calloc(size, 1);
	if (href == NULL) {
		return CRFERR_OUTOFMEMORY;
	}

	/* Store the current offset position to the file header. */
	writer->header.off_labelrefs = (uint32_t)ftell(fp);
	fseek(fp, size, SEEK_CUR);

	/* Fill members in the feature reference header. */
	strncpy(href->chunk, CHUNK_LABELREF, 4);
	href->size = 0;
	href->num = num_labels;

	writer->href = href;
	writer->state = WSTATE_LABELREFS;
	return 0;
}

int crf1mmw_close_labelrefs(crf1mmw_t* writer)
{
	FILE *fp = writer->fp;
	featureref_header_t* href = writer->href;
	uint32_t begin = writer->header.off_labelrefs, end = 0;

	/* Make sure that we are writing label feature references. */
	if (writer->state != WSTATE_LABELREFS) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Store the current offset position. */
	end = (uint32_t)ftell(fp);

	/* Compute the size of this chunk. */
	href->size = (end - begin);

	/* Write the chunk header and offset array. */
	fseek(fp, begin, SEEK_SET);
	fwrite(href, sizeof(featureref_header_t) + sizeof(uint32_t) * (href->num-1), 1, fp);

	/* Move the file pointer to the tail. */
	fseek(fp, end, SEEK_SET);

	/* Uninitialize. */
	free(href);
	writer->href = NULL;
	writer->state = WSTATE_NONE;
	return 0;
}

int crf1mmw_put_labelref(crf1mmw_t* writer, int lid, const feature_refs_t* ref, int *map)
{
	int i, fid;
	uint32_t n = 0, offset = 0;
	FILE *fp = writer->fp;
	featureref_header_t* href = writer->href;

	/* Make sure that we are writing label feature references. */
	if (writer->state != WSTATE_LABELREFS) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Store the current offset to the offset array. */
	href->offsets[lid] = ftell(fp);

	/* Count the number of references to active features. */
	for (i = 0;i < ref->num_features;++i) {
		if (0 <= map[ref->fids[i]]) ++n;
	}

	/* Write the feature reference. */
	write_uint32(fp, (uint32_t)n);
	for (i = 0;i < ref->num_features;++i) {
		fid = map[ref->fids[i]];
		if (0 <= fid) write_uint32(fp, (uint32_t)fid);
	}

	return 0;
}

int crf1mmw_open_attrrefs(crf1mmw_t* writer, int num_attrs)
{
	FILE *fp = writer->fp;
	featureref_header_t* href = NULL;
	size_t size = sizeof(featureref_header_t) + sizeof(uint32_t) * (num_attrs-1);

	/* Check if we aren't writing anything at this moment. */
	if (writer->state != WSTATE_NONE) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Allocate a feature reference array. */
	href = (featureref_header_t*)calloc(size, 1);
	if (href == NULL) {
		return CRFERR_OUTOFMEMORY;
	}

	/* Store the current offset position to the file header. */
	writer->header.off_attrrefs = (uint32_t)ftell(fp);
	fseek(fp, size, SEEK_CUR);

	/* Fill members in the feature reference header. */
	strncpy(href->chunk, CHUNK_ATTRREF, 4);
	href->size = 0;
	href->num = num_attrs;

	writer->href = href;
	writer->state = WSTATE_ATTRREFS;
	return 0;
}

int crf1mmw_close_attrrefs(crf1mmw_t* writer)
{
	FILE *fp = writer->fp;
	featureref_header_t* href = writer->href;
	uint32_t begin = writer->header.off_attrrefs, end = 0;

	/* Make sure that we are writing attribute feature references. */
	if (writer->state != WSTATE_ATTRREFS) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Store the current offset position. */
	end = (uint32_t)ftell(fp);

	/* Compute the size of this chunk. */
	href->size = (end - begin);

	/* Write the chunk header and offset array. */
	fseek(fp, begin, SEEK_SET);
	fwrite(href, sizeof(featureref_header_t) + sizeof(uint32_t) * (href->num-1), 1, fp);

	/* Move the file pointer to the tail. */
	fseek(fp, end, SEEK_SET);

	/* Uninitialize. */
	free(href);
	writer->href = NULL;
	writer->state = WSTATE_NONE;
	return 0;
}

int crf1mmw_put_attrref(crf1mmw_t* writer, int aid, const feature_refs_t* ref, int *map)
{
	int i, fid;
	uint32_t n = 0, offset = 0;
	FILE *fp = writer->fp;
	featureref_header_t* href = writer->href;

	/* Make sure that we are writing attribute feature references. */
	if (writer->state != WSTATE_ATTRREFS) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Store the current offset to the offset array. */
	href->offsets[aid] = ftell(fp);

	/* Count the number of references to active features. */
	for (i = 0;i < ref->num_features;++i) {
		if (0 <= map[ref->fids[i]]) ++n;
	}

	/* Write the feature reference. */
	write_uint32(fp, (uint32_t)n);
	for (i = 0;i < ref->num_features;++i) {
		fid = map[ref->fids[i]];
		if (0 <= fid) write_uint32(fp, (uint32_t)fid);
	}

	return 0;
}

int crf1mmw_open_features(crf1mmw_t* writer)
{
	FILE *fp = writer->fp;
	feature_header_t* hfeat = NULL;

	/* Check if we aren't writing anything at this moment. */
	if (writer->state != WSTATE_NONE) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Allocate a feature chunk header. */
	hfeat = (feature_header_t*)calloc(sizeof(feature_header_t), 1);
	if (hfeat == NULL) {
		return CRFERR_OUTOFMEMORY;
	}

	writer->header.off_features = (uint32_t)ftell(fp);
	fseek(fp, sizeof(feature_header_t), SEEK_CUR);

	strncpy(hfeat->chunk, CHUNK_FEATURE, 4);
	writer->hfeat = hfeat;

	writer->state = WSTATE_FEATURES;
	return 0;
}

int crf1mmw_close_features(crf1mmw_t* writer)
{
	FILE *fp = writer->fp;
	feature_header_t* hfeat = writer->hfeat;
	uint32_t begin = writer->header.off_features, end = 0;

	/* Make sure that we are writing attribute feature references. */
	if (writer->state != WSTATE_FEATURES) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* Store the current offset position. */
	end = (uint32_t)ftell(fp);

	/* Compute the size of this chunk. */
	hfeat->size = (end - begin);

	/* Write the chunk header and offset array. */
	fseek(fp, begin, SEEK_SET);
	fwrite(hfeat, sizeof(feature_header_t), 1, fp);

	/* Move the file pointer to the tail. */
	fseek(fp, end, SEEK_SET);

	/* Uninitialize. */
	free(hfeat);
	writer->hfeat = NULL;
	writer->state = WSTATE_NONE;
	return 0;
}

int crf1mmw_put_feature(crf1mmw_t* writer, int fid, const crf1mm_feature_t* f)
{
	FILE *fp = writer->fp;
	feature_header_t* hfeat = writer->hfeat;

	/* Make sure that we are writing attribute feature references. */
	if (writer->state != WSTATE_FEATURES) {
		return CRFERR_INTERNAL_LOGIC;
	}

	/* We must put features #0, #1, ..., #(K-1) in this order. */
	if (fid != hfeat->num) {
		return CRFERR_INTERNAL_LOGIC;
	}

	write_uint32(fp, f->type);
	write_uint32(fp, f->src);
	write_uint32(fp, f->dst);
	write_float(fp, f->weight);
	++hfeat->num;
	return 0;
}

crf1mm_t* crf1mm_new(const char *filename)
{
	FILE *fp = NULL;
	uint8_t* p = NULL;
	crf1mm_t *model = NULL;
	header_t *header = NULL;

	model = (crf1mm_t*)calloc(1, sizeof(crf1mm_t));
	if (model == NULL) {
		goto error_exit;
	}

	fp = fopen(filename, "rb");
	if (fp == NULL) {
		goto error_exit;
	}

	fseek(fp, 0, SEEK_END);
	model->size = (uint32_t)ftell(fp);
	fseek(fp, 0, SEEK_SET);

	model->buffer = (uint8_t*)malloc(model->size);
	fread(model->buffer, 1, model->size, fp);
	fclose(fp);

	/* Write the file header. */
	header = (header_t*)calloc(1, sizeof(header_t));

	p = model->buffer;
	p += read_uint8_array(p, header->magic, sizeof(header->magic));
	p += read_uint32(p, &header->size);
	p += read_uint8_array(p, header->type, sizeof(header->type));
	p += read_uint32(p, &header->version);
	p += read_uint32(p, &header->num_features);
	p += read_uint32(p, &header->num_labels);
	p += read_uint32(p, &header->num_attrs);
	p += read_uint32(p, &header->off_features);
	p += read_uint32(p, &header->off_labels);
	p += read_uint32(p, &header->off_attrs);
	p += read_uint32(p, &header->off_labelrefs);
	p += read_uint32(p, &header->off_attrrefs);
	model->header = header;

	model->labels = cqdb_reader(
		model->buffer + header->off_labels,
		model->size - header->off_labels
		);

	model->attrs = cqdb_reader(
		model->buffer + header->off_attrs,
		model->size - header->off_attrs
		);

	return model;

error_exit:
	if (model != NULL) {
		free(model);
	}
	if (fp != NULL) {
		fclose(fp);
	}
	return NULL;
}

void crf1mm_close(crf1mm_t* model)
{
	if (model->labels != NULL) {
		cqdb_delete(model->labels);
	}
	if (model->attrs != NULL) {
		cqdb_delete(model->attrs);
	}
	free(model);
}

int crf1mm_get_num_attrs(crf1mm_t* model)
{
	return model->header->num_attrs;
}

int crf1mm_get_num_labels(crf1mm_t* model)
{
	return model->header->num_labels;
}

const char *crf1mm_to_label(crf1mm_t* model, int lid)
{
	if (model->labels != NULL) {
		return cqdb_to_string(model->labels, lid);
	} else {
		return NULL;
	}
}

int crf1mm_to_lid(crf1mm_t* model, const char *value)
{
	if (model->labels != NULL) {
		return cqdb_to_id(model->labels, value);
	} else {
		return -1;
	}
}

int crf1mm_to_aid(crf1mm_t* model, const char *value)
{
	if (model->attrs != NULL) {
		return cqdb_to_id(model->attrs, value);
	} else {
		return -1;
	}
}

const char *crf1mm_to_attr(crf1mm_t* model, int aid)
{
	if (model->attrs != NULL) {
		return cqdb_to_string(model->attrs, aid);
	} else {
		return NULL;
	}
}

int crf1mm_get_labelref(crf1mm_t* model, int lid, feature_refs_t* ref)
{
	uint32_t page = model->header->off_labelrefs;
	featureref_header_t* href = (featureref_header_t*)(model->buffer + page);
	uint32_t offset = href->offsets[lid];
	ref->num_features = *(uint32_t*)(model->buffer + offset);
	ref->fids = (int*)(model->buffer + offset + sizeof(uint32_t));
	return 0;
}

int crf1mm_get_attrref(crf1mm_t* model, int aid, feature_refs_t* ref)
{
	uint32_t page = model->header->off_attrrefs;
	featureref_header_t* href = (featureref_header_t*)(model->buffer + page);
	uint32_t offset = href->offsets[aid];
	ref->num_features = *(uint32_t*)(model->buffer + offset);
	ref->fids = (int*)(model->buffer + offset + sizeof(uint32_t));
	return 0;
}

int crf1mm_get_feature(crf1mm_t* model, int fid, crf1mm_feature_t* f)
{
	uint8_t *p = NULL;
	uint32_t val = 0;
	uint32_t offset = model->header->off_features + sizeof(feature_header_t);
	offset += (sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(floatval_t)) * fid;
	p = model->buffer + offset;
	p += read_uint32(p, &val);
	f->type = val;
	p += read_uint32(p, &val);
	f->src = val;
	p += read_uint32(p, &val);
	f->dst = val;
	p += read_float(p, &f->weight);
	return 0;
}

void crf1mm_dump(crf1mm_t* crf1mm, FILE *fp)
{
	int j;
	uint32_t i;
	feature_refs_t refs;
	const header_t* hfile = crf1mm->header;

	/* Dump the file header. */
	fprintf(fp, "FILEHEADER = {\n");
	fprintf(fp, "  magic: %c%c%c%c\n",
		hfile->magic[0], hfile->magic[1], hfile->magic[2], hfile->magic[3]);
	fprintf(fp, "  size: %d\n", hfile->size);
	fprintf(fp, "  type: %c%c%c%c\n",
		hfile->type[0], hfile->type[1], hfile->type[2], hfile->type[3]);
	fprintf(fp, "  version: %d\n", hfile->version);
	fprintf(fp, "  num_features: %d\n", hfile->num_features);
	fprintf(fp, "  num_labels: %d\n", hfile->num_labels);
	fprintf(fp, "  num_attrs: %d\n", hfile->num_attrs);
	fprintf(fp, "  off_features: 0x%X\n", hfile->off_features);
	fprintf(fp, "  off_labels: 0x%X\n", hfile->off_labels);
	fprintf(fp, "  off_attrs: 0x%X\n", hfile->off_attrs);
	fprintf(fp, "  off_labelrefs: 0x%X\n", hfile->off_labelrefs);
	fprintf(fp, "  off_attrrefs: 0x%X\n", hfile->off_attrrefs);
	fprintf(fp, "}\n");
	fprintf(fp, "\n");

	/* Dump the labels. */
	fprintf(fp, "LABELS = {\n");
	for (i = 0;i < hfile->num_labels;++i) {
		const char *str = crf1mm_to_label(crf1mm, i);
#if 0
		int check = crf1mm_to_lid(crf1mm, str);
		if (i != check) {
			fprintf(fp, "WARNING: inconsistent label CQDB\n");
		}
#endif
		fprintf(fp, "  %5d: %s\n", i, str);
	}
	fprintf(fp, "}\n");
	fprintf(fp, "\n");

	/* Dump the attributes. */
	fprintf(fp, "ATTRIBUTES = {\n");
	for (i = 0;i < hfile->num_attrs;++i) {
		const char *str = crf1mm_to_attr(crf1mm, i);
#if 0
		int check = crf1mm_to_aid(crf1mm, str);
		if (i != check) {
			fprintf(fp, "WARNING: inconsistent attribute CQDB\n");
		}
#endif
		fprintf(fp, "  %5d: %s\n", i, str);
	}
	fprintf(fp, "}\n");
	fprintf(fp, "\n");

	/* Dump the transition features. */
	fprintf(fp, "TRANSITIONS = {\n");
	for (i = 0;i < hfile->num_labels;++i) {
		crf1mm_get_labelref(crf1mm, i, &refs);
		for (j = 0;j < refs.num_features;++j) {
			crf1mm_feature_t f;
			int fid = refs.fids[j];
			const char *from = NULL, *to = NULL;

			crf1mm_get_feature(crf1mm, fid, &f);
			from = crf1mm_to_label(crf1mm, f.src);
			to = crf1mm_to_label(crf1mm, f.dst);
			fprintf(fp, "  (%d) %s --> %s: %f\n", f.type, from, to, f.weight);
		}
	}
	fprintf(fp, "}\n");
	fprintf(fp, "\n");

	/* Dump the transition features. */
	fprintf(fp, "TRANSITIONS_FROM_BOS = {\n");
	crf1mm_get_labelref(crf1mm, hfile->num_labels, &refs);
	for (j = 0;j < refs.num_features;++j) {
		crf1mm_feature_t f;
		int fid = refs.fids[j];
		const char *to = NULL;

		crf1mm_get_feature(crf1mm, fid, &f);
		to = crf1mm_to_label(crf1mm, f.dst);
		fprintf(fp, "  (%d) BOS --> %s: %f\n", f.type, to, f.weight);
	}
	fprintf(fp, "}\n");
	fprintf(fp, "\n");

	/* Dump the transition features. */
	fprintf(fp, "TRANSITIONS_TO_EOS = {\n");
	crf1mm_get_labelref(crf1mm, hfile->num_labels+1, &refs);
	for (j = 0;j < refs.num_features;++j) {
		crf1mm_feature_t f;
		int fid = refs.fids[j];
		const char *from = NULL;

		crf1mm_get_feature(crf1mm, fid, &f);
		from = crf1mm_to_label(crf1mm, f.src);
		fprintf(fp, "  (%d) %s --> EOS: %f\n", f.type, from, f.weight);
	}
	fprintf(fp, "}\n");
	fprintf(fp, "\n");

	/* Dump the transition features. */
	fprintf(fp, "STATE_FEATURES = {\n");
	for (i = 0;i < hfile->num_attrs;++i) {
		crf1mm_get_attrref(crf1mm, i, &refs);
		for (j = 0;j < refs.num_features;++j) {
			crf1mm_feature_t f;
			int fid = refs.fids[j];
			const char *attr = NULL, *to = NULL;

			crf1mm_get_feature(crf1mm, fid, &f);
#if 0
			if (f.src != i) {
				fprintf(fp, "WARNING: an inconsistent attribute reference.\n");
			}
#endif
			attr = crf1mm_to_attr(crf1mm, f.src);
			to = crf1mm_to_label(crf1mm, f.dst);
			fprintf(fp, "  (%d) %s --> %s: %f\n", f.type, attr, to, f.weight);
		}
	}
	fprintf(fp, "}\n");
	fprintf(fp, "\n");
}
