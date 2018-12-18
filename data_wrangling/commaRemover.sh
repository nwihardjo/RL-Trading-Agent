#!/bin/bash

read -r -p "enter the company code: " company
echo "Removing inconsistent commas in the csv files..."

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

#echo "Adding spaces in the inconsistent commas position..."
#sed -r -e 's/^.{10}/& /' ./Processed/${company}_cleaned.staging > ./Processed/${company}_cleaned.csv

#rm -f ./Processed/${company}_cleaned.staging
echo "Inconsistent commas removed, check the header in the ./Processed/${company}_cleaned.csv"
cat ./Processed/${company}_cleaned.csv
