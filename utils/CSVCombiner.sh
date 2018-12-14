#!/bin/bash

read -r -p 'Company code : ' company
read -r -p 'Company extended code : ' addCompany

#addompany = "HK0 $company"
#company = HK1299
#addCompany = HK01299

echo "Reading company: $company $addCompany"


if [ -f ./Processed/${company}.csv ]; 
then
	echo "The compiled csv for $company has already existed! Check ./Processed/"
else
	head -1 ./2016_Data/01/${company}.csv > ./Processed/${company}.csv

	for file in ./*/*/HK/${company}.csv
	do
		echo "PIPELINING FROM  ./*/*/HK/ $files"
		tail -n +2 -q $file >> ./Processed/${company}.csv | cat
	done

	for file in ./*/*/${company}.csv
	do
		echo "PIPELINING FROM  ./*/*/ $files"
		tail -n +2 -q $file >> ./Processed/${company}.csv | cat
	done

	for file in ./*/*/${addCompany}.csv
	do
		echo "PIPELINING FROM ./*/*/ 00 $files"
		tail -n +2 -q $file >> ./Processed/${company}.csv | cat
		#tail -n +2 -q $file | cat
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
done < "./Processed/${company}.csv" > ./Processed/${company}_cleaned.csv

echo "Inconsistent commas removed, check the header: ./Processed/${company}_cleaned.csv"
cat ./Processed/${company}_cleaned.csv
