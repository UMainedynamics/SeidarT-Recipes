#!/bin/bash 

SRC_DIR="src/seidart-recipes"
DEST_DIR="docs/source" 

# Copy the readme file in the root directory to the docs/source folder 
cp README.rst docs/source 

# Copy all of the source files from the subfolders in seidart-recipes into the
# docs/source folder
for dir in "$SRC_DIR"/*/; do 
    # Check if it is a directory 
    if [ -d "$dir" ]; then 
        cd "$dir" || continue 
    
        for rst_file in *.rst; do 
            if [ -e "$rst_file" ]; then 
                echo "Copying $rst_file to $DEST_DIR"
                cp $rst_file ../../../$DEST_DIR 
            fi 
        done 
        
        cd - > /dev/null 
    fi 
done 

echo $pwd 
cd docs

sphinx-apidoc -o ./source/ ../src -e
make html 
