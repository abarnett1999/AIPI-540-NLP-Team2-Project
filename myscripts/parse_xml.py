""" Extract impressions from each x-ray report xml file. """

import xml.etree.ElementTree as et
import os
import csv


def extract_impression(doc_root):
    """ Extract only <AbstractText Label="IMPRESSION"> from xml document """

    abstract = doc_root.find('MedlineCitation/Article/Abstract')

    for abstract_text in abstract:
        label = abstract_text.get('Label')
        # print('label: ', label)
        if label == "IMPRESSION":
            return abstract_text.text


# open the file and write the header
with open('./data/impressions.csv', 'w', encoding='UTF8', newline='') as ff:
    # create the csv writer
    writer = csv.writer(ff)

    # write the header
    header = ['filename', 'impression']
    writer.writerow(header)

data_dir = "./data/xml_data/"
for fn in os.listdir(data_dir):
    f = data_dir + fn
    # checking if it is a file
    if os.path.isfile(f):
        mytree = et.parse(f)
        root = mytree.getroot()

        # extract the impression from the xml file
        impression = extract_impression(root)

        # check if impression is blank
        if impression is not None:

            # write the filename and impression to csv file
            with open('./data/impressions.csv', 'a', encoding='UTF8', newline='') as ff:
                # create the csv writer
                writer = csv.writer(ff)

                # write the data  to the csv file
                data = [fn, impression]
                # print(data)
                writer.writerow(data)
