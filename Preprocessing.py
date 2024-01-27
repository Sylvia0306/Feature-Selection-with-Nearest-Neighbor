import random 
 

def shuffle_data( input_file, output_file, number_of_lines ):

    # Open the file in read mode and read all the lines into a list 
    with open(input_file, 'r') as f: 
        lines = f.readlines() 

    # shuffle lines
    random.shuffle(lines)

    # Open the output_file in write mode and write the shuffled lines back into the file
    with open(output_file, 'w') as f: 
        f.writelines(lines[:number_of_lines]) 


def readFile( filename ) :
    # read the file and return it as list
    # note that the data is the float-point numbers
    data = []
    itemList = []
    f = open( filename, 'r' )
    for line in f :
        for item in line.split() :
            itemList.append( item )
        data.append( itemList )
        itemList = []
    f.close()
    return data


input_file = "CS170_XXXlarge_Data__5.txt"
output_file = "test_XXXL_data.txt"

# number of lines you want to keep in the output file 
number_of_lines = 1000

shuffle_data(input_file, output_file, number_of_lines)

filename = "test_XXXL_data.txt"
dataset = readFile( filename )


print(dataset)




