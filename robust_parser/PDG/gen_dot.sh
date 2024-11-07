#!/bin/bash

# Step 1: Change to the directory where the script is located  
all_c_file_path="../datasets/linux_kernel/linux-6.4-rc2-sepFile2function-withAstPdg"
temp_dot_file="temp-dot"
joern_path="../joern/joern-cli"
count_c=0

cd $all_c_file_path

# prepre:delete some files
rm -rf $all_c_file_path/$temp_dot_file
rm cpg.bin

# Step 2: Loop through the directory and find all .c files  
for file in */*.c; do

   ((count_c++))

   echo -e "\n\n\n\n****************start joern****************\n\n\n\n"
   echo -e "c file path: $all_c_file_path/$file\n"
   echo -e "c file count: $count_c\n"

   folder_path=$(dirname "$file")
   $joern_path/joern-parse $folder_path

   echo -e "temp dot path: $all_c_file_path/$temp_dot_file\n"

   $joern_path/joern-export cpg.bin --repr pdg --out $temp_dot_file

   # Step 2.2: Delete the generated cpg.bin file  
   rm cpg.bin
   echo -e "Delete the generated cpg.bin file  \n"

   # Step 2.3: Find the largest .dot file in the directory and move it to the .c file's directory 
   echo -e "Find the largest .dot file in the directory and move it to the .c file's directory \n"
   dot_files=$(ls $temp_dot_file/*.dot)
   max_file_size=0
   max_file_path=""
   for dot_file in $dot_files
   do
      file_size=$(stat -c%s $dot_file)
      if [ $file_size -gt $max_file_size ]; then
         max_file_size=$file_size
         max_file_path=$dot_file
      fi
   done
   echo -e "max dot file path: $max_file_path\n"  
   mv "$max_file_path" "$(dirname "$file")/$(basename "$file").dot"
   rm -rf $all_c_file_path/$temp_dot_file
   echo -e "\n\n\n\n****************finish joern****************\n\n\n\n"
done
