#!/usr/bin/perl

use lib './blib/lib';
use lib './blib/arch';
use CRFSuite;
use strict;

our $DEBUG = 0;

my $modelfile = shift @ARGV;

my $trainer = CRFSuite::Trainer->new();

sub CRFSuite::Trainer::message {
    # FIXME: This function does not work, because message callback
    # function has not been implemented.  About perl callback
    # function, see http://www.swig.org/papers/Perl98/swigperl.htm.
    my $this = shift;
    print @_;
}

my $xseq = CRFSuite::ItemSequence->new();
my $yseq = CRFSuite::StringList->new();
while (<>) {
    chomp;
    unless( $_ ){
	# An empty line presents an end of a sequence.
	$trainer->append( $xseq, $yseq, 0 );
	if ($DEBUG) {
	    for ( my $i = 0; $i < $xseq->size; $i++ ) {
		my $x = $xseq->get($i);
		printf "%s\t", $yseq->get($i);
		for ( my $j = 0; $j < $x->size; $j++ ) {
		    my $f = $x->get($j);
		    printf "\t%s:%d", $f->swig_attr_get, $f->swig_value_get;
		}
		print "\n";
	    }
	}
	$xseq = CRFSuite::ItemSequence->new();
	$yseq = CRFSuite::StringList->new();
    } else {
	# Split the line with TAB characters.
	my( $label, @field ) = split( /\t/, $_ );
	# Append attributes to the item.
	my $item = CRFSuite::Item->new();
	for my $x ( @field ) {
	    if ( $x =~ s/:([\.\d]+)\Z// ) {
		# Weighted attribute
		$item->push( CRFSuite::Attribute->new( $x, $1+0 ) );
	    } else {
		# Unweighted (weight=1) attribute.
		$item->push( CRFSuite::Attribute->new( $x ) );
	    }
	}
	# Append the item to the item sequence.
	$xseq->push( $item );
	# Append the label to the label sequence.
	$yseq->push( $label );
    }
}

# Use L2-regularized SGD and 1st-order dyad features.
$trainer->select('l2sgd', 'crf1d');

# Set the coefficient for L2 regularization to 0.1
$trainer->set('c2', '0.1');

# This demonstrates how to list parameters and obtain their values.
for my $name ( @{$trainer->params} ) {
    printf <<__format__, $name, $trainer->get($name), $trainer->help($name);
parameter: %s
value: %s
help: %s
__format__
}

# Start training; the training process will invoke trainer.message()
# to report the progress.
$trainer->train($modelfile, -1);
