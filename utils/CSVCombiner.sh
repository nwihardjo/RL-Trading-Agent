#!/bin/bash
# need to check the absolute path of each directory

read -r -p 'Company code with no space (i.e. HK0102): ' company
read -r -p 'Company extended code with no space (i.e. HK0102 -> HK00102, add extra 0 following HK): ' addCompany


#TODO: integrate absolute path in the variable declaration
stagingPath="../new_data/.staging/"
rawPath="../new_data/Raw/*/*/"


echo "Reading company: $company $addCompany"

if [ -f ../new_data/.staging/${company}_cleaned.csv ]; 
then
	echo "The compiled csv for $company has already existed! Check ../new_data/.staging/"
else
	head -1 ../new_data/Raw/2016_Data/01/${company}.csv > ../new_data/.staging/${company}.csv

	# */*/ is for (i.e.) 2015_Data/HK_1Min_201501
	for file in ../new_data/Raw/*/*/HK/${company}.csv
	do
		echo "PIPELINING FROM  ../new_data/Raw/*/*/HK/ $files"
		tail -n +2 -q $file >> ../new_data/.staging/${company}.csv | cat
	done

	# */*/ is for (i.e.) 2016_Data/05
	for file in ../new_data/Raw/*/*/${company}.csv
	do
		echo "PIPELINING FROM  ../new_data/Raw/*/*/ $files"
		tail -n +2 -q $file >> ../new_data/.staging/${company}.csv | cat
	done

	for file in ../new_data/Raw/*/*/${addCompany}.csv
	do
		echo "PIPELINING FROM ../new_data/Raw/*/*/ 00 $files"
		tail -n +2 -q $file >> ../new_data/.staging/${company}.csv | cat
	done
fi


echo "Removing inconsistent comma in CSV files..."

while read -r line 
do
	if [ "${line:0:10}" == "2004-00-00" ] || [ "${line:0:4}" == "2014" ]; then
		continue
	elif [ "${line:10:1}" == "," ]; then
		temp="${line/,/}"
		echo "${temp:0:10} ${temp[@]:10}"
	else
		echo "${line}"
	fi
done < "../new_data/.staging/${company}.csv" > ../new_data/.staging/${company}_cleaned.csv


echo "Inconsistent commas removed, check the header: ../new_data/.staging/${company}_cleaned.csv"
cat ../new_data/.staing/${company}_cleaned.csv
rm -rf ../new_data/.staging/${company}.csv
