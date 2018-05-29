##############################################################################
# Claudia Schulz
# November 2017
#
# for some embeddings the first line may have to be removed
##############################################################################

# USER ACTION NEEDED
# specify embedding file to process
embeddingFile = ""
embeddingNew = embeddingFile + "_new"

with open(embeddingFile, "r") as input, open(embeddingNew, "w") as output:
    lines = input.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            print(line)
        else:
            output.write(line)
