#!/bin/bash

rm -r ../datasets/ARCH # remove any old version
mkdir ../datasets/ARCH
cd ../datasets/ARCH

wget https://warwick.ac.uk/fac/cross_fac/tia/data/arch/books_set.zip
unzip books_set.zip
rm -r __MACOSX

wget https://warwick.ac.uk/fac/cross_fac/tia/data/arch/pubmed_set.zip
unzip pubmed_set.zip
rm -r __MACOSX

# remove both zip files
rm ../datasets/ARCH/*.zip
