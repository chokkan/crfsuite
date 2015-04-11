require 'mkmf'
$LOCAL_LIBS +=' -lcrfsuite'
dir_config('crfsuite')
create_makefile('crfsuite')
