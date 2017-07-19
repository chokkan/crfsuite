#!/usr/bin/env ruby

require 'crfsuite'

# Inherit Crfsuite::Trainer to implement message() function, which receives
# progress messages from a training process.
class Trainer < Crfsuite::Trainer
  def message(s)
    # Simply output the progress messages to STDOUT.
    print s
  end
end

def instances(fi)
  xseq = Crfsuite::ItemSequence.new
  yseq = Crfsuite::StringList.new

  Enumerator.new do |y|
    fi.each do |line|
      line.chomp!
      if line.empty?
        # An empty line presents an end of a sequence.
        y << [xseq, yseq]
        xseq = Crfsuite::ItemSequence.new
        yseq = Crfsuite::StringList.new
        next
      end

      # Split the line with TAB characters.
      fields = line.split("\t")

      # Append attributes to the item.
      item = Crfsuite::Item.new
      fields.drop(1).each do |field|
        idx = field.rindex(':')
        if idx.nil?
          # Unweighted (weight=1) attribute.
          item << Crfsuite::Attribute.new(field)
        else
          # Weighted attribute
          item << Crfsuite::Attribute.new(field[0..idx-1], field[idx+1..-1].to_f)
        end
      end

      # Append the item to the item sequence.
      xseq << item
      # Append the label to the label sequence.
      yseq << fields[0]
    end
  end
end

if __FILE__ == $0
  model_file = ARGV[0]

  # This demonstrates how to obtain the version string of CRFsuite.
  puts Crfsuite.version

  # Create a Trainer object.
  trainer = Trainer.new

  # Read training instances from STDIN, and set them to trainer.
  instances($stdin).each do |xseq, yseq|
    trainer.append(xseq, yseq, 0)
  end

  # Use L2-regularized SGD and 1st-order dyad features.
  trainer.select('l2sgd', 'crf1d')

  # This demonstrates how to list parameters and obtain their values.
  trainer.params.each do |name|
    print name, ' ', trainer.get(name), ' ', trainer.help(name), "\n"
  end

  # Set the coefficient for L2 regularization to 0.1
  trainer.set('c2', '0.1')

  # Start training; the training process will invoke trainer.message()
  # to report the progress.
  trainer.train(model_file, -1)
end
