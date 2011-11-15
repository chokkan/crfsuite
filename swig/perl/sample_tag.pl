#!/usr/bin/perl

use lib './blib/lib';
use lib './blib/arch';
use CRFSuite;
use strict;

# Create a tagger object.
my $tagger = CRFSuite::Tagger->new();

# Load the model to the tagger.
$tagger->open( shift @ARGV );

my $xseq = CRFSuite::ItemSequence->new();
while (<>) {
    chomp;
    unless( $_ ){
	# An empty line presents an end of a sequence.
    	# Tag the sequence.
        $tagger->set($xseq);
        # Obtain the label sequence predicted by the tagger.
        my $yseq = $tagger->viterbi();
	# Output the probability of the predicted label sequence.
        printf "%f\n", $tagger->probability($yseq);
	for( my $i = 0; $i <= $#{$yseq}; $i++ ){
	    # Output the predicted labels with their marginal probabilities.
	    printf "%s:%f\n", $yseq->[$i], $tagger->marginal($yseq->[$i], $i);
	}
    } else {
	# Split the line with TAB characters.
	my( undef, @field ) = split( /\t/, $_ );
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
    }
}
