def parse_text(file_name):
    file_object = open(file_name, "r")
    files, objects = list(), list()
    NUM_OBJECTS = 0
    for line in file_object:
        words = line.split()
        for word in words:
            word_split = word.split('.')
            for new_word in word_split:
                if new_word == 'jpg':
                    files.append(word)
        if words[0] == 'Objects':
            objects_line = list()
            for word in words:
                if word != 'Objects' and word != 'Detected:':
                    objects_line.append(word)
                    NUM_OBJECTS += 1
            objects.append(objects_line)

    return dict(zip(files, objects))


file1 = "000annotations.txt"
file2 = "object_out_all_blurs_multiblur_NN.txt"
file3 = "object_out_all_blurs_nonmultiblur_NN.txt"
file4 = "object_out_all_blurs_tom_NN.txt"

original = parse_text(file1)
multiblur = parse_text(file2)
nonmultiblur = parse_text(file3)
tom = parse_text(file4)




types = ['nearest', 'sinc', 'linear', 'cubic', 'area']

n_misses = dict()
for type in types:
    n_misses[type] = 0
    for key in original.keys():
        originally_found= set(original[key])
        try:
            newly_found = set(tom[type+key])
            diff = originally_found.difference(newly_found)
        except KeyError:
            diff = originally_found
            #print(diff)
        n_misses[type] += len(diff)

print(n_misses)


