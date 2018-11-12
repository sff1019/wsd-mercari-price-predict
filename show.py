import csv


tsv_file = open('mercari/train.tsv', 'r')
tsv = list(csv.reader(tsv_file, delimiter='\t'))

for line in tsv:
    print(line)
