#!/bin/bash

if ! command -v compare_larcv2_files.py &> /dev/null
then
    echo -e "\033[5;1;33;91m[TEST FAILED]\033[00m : compare_larcv2_files.py could not be found (check if your larcv2 installation is up to date)"
    exit 1
fi

SOURCE_FILE=unit_test_flow_2x2.h5
REFERENCE_FILE=unit_test_larcv_2x2.root
OUTPUT_FILE=supera-$$.root

SOURCE_FILE_ID=1gzYIs5a8WPngHjUOfphtbGrrkdN9mWqP
REFERENCE_FILE_ID=1Cb6YsZj8xQITZvwICGZZu9ZMK-Av58AE

if [ -f $OUTPUT_FILE ]; then
    echo -e "\033[5;1;33;91m[TEST FAILED]\033[00m : The larcv file to be created already exist: ${OUTPUT_FILE}"
    echo "Please (re)move the file and try again."
    exit;
fi

echo
echo "Downloading flow file to run supera"
echo gdown -O $SOURCE_FILE $SOURCE_FILE_ID
gdown -O $SOURCE_FILE $SOURCE_FILE_ID
if [ ! -f $SOURCE_FILE ]; then
    echo -e "\033[5;1;33;91m[TEST FAILED]\033[00m : Failed to download the source flow file from https://drive.google.com/file/d/${SOURCE_FILE_ID}/view?usp=drive_link";
    exit;
fi

echo
echo "Downloading larcv file to compare against"
echo gdown -O $REFERENCE_FILE $REFERENCE_FILE_ID
gdown -O $REFERENCE_FILE $REFERENCE_FILE_ID
if [ ! -f $REFERENCE_FILE ]; then
    echo -e "\033[5;1;33;91m[TEST FAILED]\033[00m : Failed to download the reference larcv2 file from https://drive.google.com/file/d/${REFERENCE_FILE_ID}/view?usp=drive_link";
    exit;
fi

echo
echo "Running flow2supera"
echo run_flow2supera.py -c 2x2_mpvmpr -o $OUTPUT_FILE $SOURCE_FILE
run_flow2supera.py -c 2x2_mpvmpr -o $OUTPUT_FILE $SOURCE_FILE

echo
echo "Running unit test script"
echo compare_larcv2_files.py -r $REFERENCE_FILE -t $OUTPUT_FILE
compare_larcv2_files.py -r $REFERENCE_FILE -t $OUTPUT_FILE

ret=$?
if [ $ret -ne 0 ]; then
    echo
    echo -e "\033[5;1;33;91m[TEST FAILED]\033[00m : compare_larcv2_files.py returned the error code ${ret}"
    echo
else
    echo
    echo -e "\033[5;1;33;5m[TEST SUCCESS (pass)]\033[0m : supera behavior remains the same"
    echo
fi

rm -f $OUTPUT_FILE $REFERENCE_FILE $SOURCE_FILE
