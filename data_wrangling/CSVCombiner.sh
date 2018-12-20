#!/bin/bash

# this bash script is used to aggregate and clean the stock price for a particular stock into a single csv


read -r -p 'Company code (i.e. HK0102): ' company
read -r -p 'Company extended code (i.e. HK0102 -> HK00102, add extra 0 following HK): ' addCompany

# removing any spaces in the input
company="$(echo -e "${company}" | tr -d '[:space:]')"
addCompany="$(echo -e "${addCompany}" | tr -d '[:space:]')"
echo "Reading company: $company and $addCompany"

# CHANGE these absolute paths when necessary
stagingPath="../data/.staging/"
rawPath="../data/Raw/*/*/"
headerPath="../data/Raw/2016_Data/01/"

# if staging file already exist, no need to aggregate it all over again
exist=false;



# DELETE files if want to aggregate the file again. There are 3 different directories from the unzipped / uncompressed
#   data, each one of them has its for-loop. In some cases, the duplication in the data present (data of 2016 exists in
#   the 2017 file, vice versa), do double-check the data to make sure no duplication exists.
if [ -f ${stagingPath}${company}_cleaned.csv ]; 
then
	exist=true;
	echo "The compiled csv for $company has already existed! Check $stagingPath"
else
	# retrieve the column headers, ASSUMING all the files have the same columns
	head -1 ${headerPath}${company}.csv > ${stagingPath}${company}.csv

	# */*/ is for (i.e.) 2015_Data/HK_1Min_201501
	for file in ${rawPath}HK/${company}.csv
	do
		echo "Retrieving data from ${rawPath}/HK/ directory for: $files"
		# get the data without the column header
		tail -n +2 -q $file >> ${stagingPath}${company}.csv | cat
	done

	# */*/ is for (i.e.) 2016_Data/05
	for file in ${rawPath}${company}.csv
	do
		echo "Retrieving data from ${rawPath} directory for: $files"
		tail -n +2 -q $file >> ${stagingPath}${company}.csv | cat
	done

    # for file which has new format, with the following 0 following HK in the filename
	for file in ${rawPath}${addCompany}.csv
	do
		echo "Retrieving data from $rawPath directory for extended company code: $files"
		tail -n +2 -q $file >> ${rawPath}${company}.csv | cat
	done
fi



# assume if the aggregated files already exist, it already is cleaned
if [ $exist = false ]; then 
	echo "Removing inconsistent comma in CSV files..."
	
	while read -r line 
	do
	    # remove the noise in the data (has the format of 2004 without any prices, volume, etc.)
		if [ "${line:0:10}" == "2004-00-00" ] || [ "${line:0:4}" == "2014" ]; then
			continue
		# remove comma which is used to separate the date and time as it mess with the column in csv
		elif [ "${line:10:1}" == "," ]; then
			temp="${line/,/}"
			echo "${temp:0:10} ${temp[@]:10}"
		else
			echo "${line}"
		fi
	done < "${stagingPath}${company}.csv" > ${stagingPath}${company}_cleaned.csv
	
	echo "Inconsistent commas removed, check the header: ../new_data/.staging/${company}_cleaned.csv"
fi

rm -rf ../new_data/.staging/${company}.csv

# uncomment the following line of code to enable the print of the whole file to the terminal for checking whether
#   duplication exists
#cat ../new_data/.staging/${company}_cleaned.csv

