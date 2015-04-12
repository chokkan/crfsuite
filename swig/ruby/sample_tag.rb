#!/usr/bin/env ruby

require 'crfsuite'

def instances(fi)
  xseq = Crfsuite::ItemSequence.new

  Enumerator.new do |y|
    fi.each do |line|
      line.chomp!
      if line.empty?
        # An empty line presents an end of a sequence.
        y << xseq
        xseq = Crfsuite::ItemSequence.new
        next
      end

      # Split the line with TAB characters.
      fields = line.split("\t")
      item = Crfsuite::Item.new
      fields.drop(1).each do |field|
        idx = field.rindex(':')
        if idx.nil?
          # Unweighted (weight=1) attribute.
          item << Crfsuite::Attribute.new(field)
        else
          # Weighted attribute
          item << Crfsuite::Attribute.new(field[0..idx-1], float(field[idx+1..-1]))
        end
      end

      # Append the item to the item sequence.
      xseq << item
    end
  end
end

if __FILE__ == $0
  model_file = ARGV[0]

  # Create a tagger object.
  tagger = Crfsuite::Tagger.new

  # Load the model to the tagger.
  tagger.open(model_file)

  instances($stdin).each do |xseq|
    # Tag the sequence.
    tagger.set(xseq)
    # Obtain the label sequence predicted by the tagger.
    yseq = tagger.viterbi()
    # Output the probability of the predicted label sequence.
    puts tagger.probability(yseq)
    yseq.each_with_index do |y, t|
      # Output the predicted labels with their marginal probabilities.
      puts '%s:%f' % [y, tagger.marginal(y, t)]
    end
    puts ''
  end
end
