#!/usr/bin/env python3

import os
import re
import sys
import glob
import pandas as pd

from section_tagger import process_report, get_unknown_entities
from section_tagger import section_tagger_init

# XML character entities
regex_xml_character_entity = re.compile(r'&(?:#([0-9]+)|#x([0-9a-fA-F]+)|([0-9a-zA-Z]+));');

# one or more spaces and newlines
regex_multi_space   = re.compile(r' +')
regex_multi_newline = re.compile(r'\n+')

my_sent_tokenize = lambda text: text.split('\n')

labels_to_merge = {
    'ADMISSION_MEDICATIONS': 'MEDICATIONS',
    'DISCHARGE_MEDICATIONS': 'MEDICATIONS',
    'LABORATORY_AND_RADIOLOGY_DATA': 'PHYSICAL_EXAMINATION'
}

# "ALLERGIES_AND_ADVERSE_REACTIONS"
# "FAMILY_MEDICAL_HISTORY",
# "PERSONAL_AND_SOCIAL_HISTORY",
# "PAST_MEDICAL_HISTORY"

labels_to_include = ["HOSPITAL_COURSE", 
                     "PHYSICAL_EXAMINATION",
                     "HISTORY_PRESENT_ILLNESS",
                     "MEDICATIONS"]

def transform(obj, fasttext=False):
    txt = obj['file']
    
    with open(txt) as f:
        # Original text file
        report = f.read().strip('\n')
        
        '''
        sents = my_sent_tokenize(report)
        
        for sent in enumerate(sents):
            section_headers, section_text = process_report(sent)
            for h in section_headers:
                print (str(h))
                print (h.to_output_string())
        '''
         
        # process all reports sent via command line
        # replace any XML entities with a space
        no_xml_entities = regex_xml_character_entity.sub(' ', report)

        # replace repeated newlines with a single newline
        single_newline_report = regex_multi_newline.sub('\n', no_xml_entities)

        # replace repeated spaces with a single space
        clean_report = regex_multi_space.sub(' ', single_newline_report)

        # run the section tagger and print results to stdout
        section_headers, section_text = process_report(clean_report)
        
        section_dict = {}
        for i, sh in enumerate(section_headers):
            section_dict[str(sh.sentence_index)]=sh.concept
        
        sents = my_sent_tokenize(report)
        current_section = None
        skip_list = ["IGNORE", "DATE", "PATIENT_NAME", "UNIT", "REPORT", "REPORT_STATUS", "ROOM", "CODE_STATUS", "UNKNOWN", "PROVIDERS", "ATTENDING_PHYSICIAN", "SERVICE", "CHIEF_COMPLAINT", "AUTHOR", "ADMISSION_DATE", "DATE_OF_DISCHARGE", "DATE_OF_BIRTH", "GENDER", "DATE_TRANSCRIBED", "DATE_TIME", "PHONE_NUMBER", "DATE_DICTATED", "AGE", "MRN", "DICTATING_PHYSICIAN", "CODE_STATUS"]
        retStr = ""
        for lineno, sent in enumerate(sents):
            if section_dict.get(str(lineno))!=None:
                current_section = section_dict[str(lineno)]
                if current_section in skip_list:
                    current_section = "ignore"
            if sent.find(", M.D.")!=-1:
                #print (sent)
                current_section=None # skip the line contianing doctor name
            if current_section not in (None, "ignore"):
                if labels_to_merge.get(current_section)!=None:
                    current_section = labels_to_merge[current_section]
                if current_section in labels_to_include:
                    if fasttext:
                        retStr += current_section + "," + str.replace(sent, ",", " ") + "\n"
                    else:
                        retStr +=  "c=\"%s\" %s ||t=\"%s\"\n" % (sent, lineno+1, current_section)
        
        return retStr
    
def main():
    
    fasttext = True
    
    # initialize the section tagger
    if not section_tagger_init():
        sys.exit(-1)

    # now read files in training folder and apply the structure and
    # and generate the token files
    folders = {"train":"dataset/unannotated/training", "validation":"dataset/unannotated/validation"}
    file_mappings = {"train":[], "validation": []}
    for key in list(folders.keys()):
        folder = folders[key]
        print('\n\tReading from ' + folder + ' directory\n')
        for file in glob.glob(folder + "/*.txt"):
            file_name = os.path.basename(file)
            file_mappings[key].append({'file': file, 'file_name': file_name})
        
    buffer = []
        
    for key in list(file_mappings.keys()):
        folder = folders[key]
        for obj in file_mappings[key]:
            val = transform(obj, fasttext)
            if fasttext == False:
                out_path = os.path.join(folder, "output", obj['file_name'].replace(".txt", ".top"))
                sys.stdout.write('\n\nwriting to: %s\n' % out_path)
                with open(out_path, 'w') as f:
                    f.write(str('%s\n' % val))
                    sys.stdout.write('\n')
            else:
                buffer.append(val)
        if fasttext:
            out_path = os.path.join(folder, "output", 'fasttext.csv')                
            sys.stdout.write('\n\nwriting to: %s\n' % out_path)
            with open(out_path, 'w') as f:
                f.write("label,text\n")
                for buf in buffer:
                    f.write(str('%s' % buf))
                
            
    pd.DataFrame(list(get_unknown_entities().items()),columns=('entity', 'count')).to_csv("unknown.csv")
        
if __name__ == '__main__':
    main()
    