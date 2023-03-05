import numpy as np
import random
import pickle
import traceback
import xml.etree.ElementTree as etree
from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

import datasets
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext
def pullTags(raw_html):
    tags = re.findall(CLEANR, raw_html)
    tags = [tag.replace("<", "").replace(">", "").replace("-", " ") for tag in tags]
    return tags
def stackex_philosophy(data_root):
    data = {}
    qa_pairs = {}
    questions = {}
    answers = {}
    labels = []
    for event, elem in tqdm(etree.iterparse(data_root + "/Posts.xml", events=('end',)), desc="Parsing {} XML file".format("stackexchange_philosophy")):
        if elem.tag == "row":
            attribs = defaultdict(lambda: None, elem.attrib)
            if attribs["PostTypeId"] == '1':
                tags = pullTags(attribs["Tags"])
                for tag in tags:
                    if tag not in labels:
                        labels.append(tag)
                questions[int(attribs["Id"])] = {"Body": cleanhtml(attribs["Body"].replace("\n", " ")), "Tags": tags}
                if "AcceptedAnswerId" in attribs.keys():
                    qa_pairs[int(attribs["Id"])] = int(attribs["AcceptedAnswerId"])
                    if int(attribs["AcceptedAnswerId"]) in answers.keys():
                        questions[int(attribs["id"])]["Body"] += " " + answers[int(attribs["AcceptedAnswerId"])]["Body"]
            if attribs["PostTypeId"] == '2':
                answers[int(attribs["Id"])] = {"Body": cleanhtml(attribs["Body"].replace("\n", " "))}
                if int(attribs["ParentId"]) in qa_pairs.keys() and int(attribs["Id"]) == qa_pairs[int(attribs["ParentId"])]:
                    questions[int(attribs["ParentId"])]["Body"] += " " + answers[int(attribs["Id"])]["Body"]
    for key, value in questions.items():
        data[value["Body"]] = value["Tags"]
    return data, labels

    # with open(data_root + '/Posts.xml') as f:
    #     for line in f:
    #         if line.startswith("<row Id"):

def RCV1(data_root, sample=False, start_index = 0):
    label_vocab = {}
    with open(data_root + '/labels_vocab.txt') as f:
        for line in f:
            description = line.split("child-description: ")[1].strip("\n")
            label = line.split("child: ")[1].split(" ")[0]
            if description != "No Description":
                label_vocab[label] = description
    labels = {}

    with open(data_root + '/rcv1-v2.topics.qrels') as f:
        for line in f:
            label, DID, ignore = line.split(" ")
            sample_idx = int(DID)


            if sample_idx in labels.keys():
                labels[sample_idx].append(label_vocab[label])
            else:
                labels[sample_idx] = [label_vocab[label]]

    idx = 0
    label2id = {}
    for label in label_vocab.values():
        label2id[label] = int(idx)
        idx = idx + 1
    prior = np.zeros((len(label2id), len(label2id)))
    with open(data_root + '/hierarchy.txt') as f:
        for line in f:
            parent = line.split("parent: ")[1].split("child:")[0].strip()
            child = line.split("child: ")[1].split("child-description")[0].strip()
            if parent != "Root" and child != "Root":
                prior[label2id[label_vocab[parent]], label2id[label_vocab[child]]] = 1

    train = {}

    with open(data_root + '/lyrl2004_tokens_train.dat') as f:
        print("train data DIDs:")
        DID = 0

        text = ""
        for line in f:
            if ".I" in line:

                if DID != 0:
                    train[text.strip()] = labels[DID]
                    text = ""
                DID = int(line.split(".I ")[1])
            elif ".W" not in line and len(line) > 1:
                text += line.strip("\n") + " "
        train[text.strip()] = labels[DID]
    if sample is True:
        reduced = random.sample(list(train.items()), 2000)
        train_idx = random.sample(range(2000), 1000)
        test_idx = set(range(2000)) - set(train_idx)
        reduced_train = {reduced[idx][0]: reduced[idx][1] for idx in train_idx}
        reduced_test = {reduced[idx][0]: reduced[idx][1] for idx in test_idx}
        return {"train": reduced_train, "test": reduced_test, "label2id": label2id, "prior": prior}

    test = {}
    for file in ["lyrl2004_tokens_test_pt0.dat"]:
         # "lyrl2004_tokens_test_pt1.dat", "lyrl2004_tokens_test_pt2.dat", "lyrl2004_tokens_test_pt3.dat"
        with open(data_root + '/' + file) as f:
            DID = 0
            print("test DIDs:")
            text = ""
            for line in f:
                if ".I" in line:
                    if DID != 0 :
                        test[text.strip()] = labels[DID]
                        text = ""
                    DID = int(line.split(".I ")[1])
                elif ".W" not in line and len(line) > 1:

                    text += line.strip("\n") + " "

            test[text.strip()] = labels[DID]

    return {"train": train, "test": test, "label2id": label2id, "prior": prior}



def delicious():
    #available at https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels
    data_root = '/home/muberra/Desktop/PycharmProjects/XMTC/resources/DeliciousMIL/Data'
    vocab = {}
    data = {}
    train = {}
    test = {}
    test_labels = []
    train_labels = []
    label_vocab = {}
    with open(data_root + '/vocabs.txt') as f:
        for line in f:
            (val, key) = line.split(',')
            vocab[int(key)] = val
    with open(data_root + '/labels.txt') as f:
        for line in f:
            (val, key) = line.split(',')
            label_vocab[int(key)] = val
    with open(data_root + '/train-label.dat') as f:
        for line in f:
            labelText = []
            labelVector = np.nonzero([int(entry) for entry in line.split(" ") ])
            for label in labelVector[0]:
                labelText.append(label_vocab[label])
            train_labels.append(labelText)
    i = 0
    with open(data_root + '/train-data.dat') as f:
        for line in f:
            text = ""
            for word in line.split(" "):
                if word[0] != "<":
                    text += vocab[int(word)] + " "
            train[text] = train_labels[i]
            i = i + 1
    with open(data_root + '/test-label.dat') as f:
        for line in f:
            labelText = []
            labelVector = np.nonzero([int(entry) for entry in line.split(" ") ])
            for label in labelVector[0]:
                labelText.append(label_vocab[label])
            test_labels.append(labelText)
    i = 0
    with open(data_root + '/test-data.dat') as f:
        for line in f:
            text = ""
            for word in line.split(" "):
                if word[0] != "<":
                    text += vocab[int(word)] + " "
            test[text] = test_labels[i]
            i = i + 1
    return {"train": train, "test": test}, label_vocab.values()
'''
Set Reducer:
reduces the dataset for weakly supervised learning.
dataset: the full dataset
label2id: the list of all distinct labels
size: desired output dataset size
e_label: expected number of labels
k: fraction of selected documents that will have more than e_label labels
'''
def set_reducer(dataset, label2id, size, e_label, k, seed = 0):
    r = random.Random()
    excluded = []
    if seed != 0:
        r.seed(seed)
    reduced = {}
    moreLabel = []
    lessLabel = []
    for label in dataset.values():
        if len(label) > e_label:
            moreLabel.append(label)
        if len(label) < e_label:
            lessLabel.append(label)
    for label in label2id.values():
        if label == 'GROUNDNUT-OIL':
            print()
        rand = r.random()
        if rand < k:
            moreLabelOptions = [labelList for labelList in moreLabel if label in labelList]
            if len(moreLabelOptions) == 0:
                lessLabelOptions = [labelList for labelList in lessLabel if label in labelList]
                if len(lessLabelOptions) == 0:
                    print(label)
                    excluded.append(label)
                    continue
                selectedLessLabelList = r.choice(lessLabelOptions)
                choiceList = list(dataset.keys())[list(dataset.values()).index(selectedLessLabelList)]
                if isinstance(choiceList, str):
                    chosenDoc = choiceList
                else:
                    chosenDoc = r.choice(list(dataset.keys())[list(dataset.values()).index(selectedLessLabelList)])
                if chosenDoc in reduced.keys():
                    reduced[chosenDoc] = reduced[chosenDoc] + (selectedLessLabelList)
                else:
                    reduced[chosenDoc] = selectedLessLabelList
            else:
                selectedMoreLabelList = r.choice(moreLabelOptions)
                if chosenDoc in reduced.keys():
                    reduced[chosenDoc] = reduced[chosenDoc] + (selectedMoreLabelList)
                else:
                    reduced[chosenDoc] = selectedMoreLabelList
        else:
            lessLabelOptions = [labelList for labelList in lessLabel if label in labelList]
            if len(lessLabelOptions) == 0:
                moreLabelOptions = [labelList for labelList in moreLabel if label in labelList]
                if len(moreLabelOptions) == 0:
                    print(label)
                    excluded.append(label)
                    continue
                selectedMoreLabelList = r.choice(moreLabelOptions)

                choiceList = list(dataset.keys())[list(dataset.values()).index(selectedMoreLabelList)]
                if isinstance (choiceList, str):
                    chosenDoc = choiceList
                else:
                    chosenDoc = r.choice(list(dataset.keys())[list(dataset.values()).index(selectedMoreLabelList)])
                if chosenDoc in reduced.keys():
                    reduced[chosenDoc] = reduced[chosenDoc] + (selectedMoreLabelList)
                else:
                    reduced[chosenDoc] = selectedMoreLabelList
            else:
                selectedLessLabelList = r.choice(lessLabelOptions)
                choiceList = list(dataset.keys())[list(dataset.values()).index(selectedLessLabelList)]
                if isinstance(choiceList, str):
                    chosenDoc = choiceList
                else:
                    chosenDoc = r.choice(list(dataset.keys())[list(dataset.values()).index(selectedLessLabelList)])
                if chosenDoc in reduced.keys():
                    reduced[chosenDoc].append(selectedLessLabelList)
                else:
                    reduced[chosenDoc] = selectedLessLabelList
            print(len(reduced))
    while len(reduced) < size:

        randChoice = r.choice(list(dataset.keys()))
        if randChoice not in reduced.keys():
            reduced[randChoice] = dataset[randChoice]
            x = reduced[randChoice]
            #print(len(reduced))
    #print(r.random())
    return reduced
def ReutersHierarchyBuilder(topics):
    hierarchy = {}
    leaves = []
    for i in range(len(topics)):
        if i < 3:
            if i == 0:
                hierarchy["Root"] = []
            hierarchy["Root"].append(topics[i])
            leaves.append(topics[i])
        elif i < 19:
            if i == 3:
                hierarchy["Economic Indicator Codes"] = []
            hierarchy["Economic Indicator Codes"].append(topics[i])
            leaves.append(topics[i])
        elif i < 45:
            if i == 19:
                hierarchy["Currency Codes"] = []
            hierarchy["Currency Codes"].append(topics[i])
            leaves.append(topics[i])
        elif i < 47:
            if i == 45:
                hierarchy["Corporate Codes"] = []
            hierarchy["Corporate Codes"].append(topics[i])
            leaves.append(topics[i])
        elif i < 125:
            if i == 47:
                hierarchy["Commodity Codes"] = []
                hierarchy["Commodity Codes"].append("CASTORSEED")
                hierarchy["CASTORSEED"] = []
                hierarchy["Commodity Codes"].append("COCONUT")
                hierarchy["COCONUT"] = []
                hierarchy["Commodity Codes"].append("CORN")
                hierarchy["CORN"] = []
                hierarchy["Commodity Codes"].append("COTTON")
                hierarchy["COTTON"] = []
                hierarchy["Commodity Codes"].append("GROUNDNUT")
                hierarchy["GROUNDNUT"] = []
                hierarchy["Commodity Codes"].append("LINSEED")
                hierarchy["LINSEED"] = []
                hierarchy["Commodity Codes"].append("PALMKERNEL")
                hierarchy["PALMKERNEL"] = []
                hierarchy["Commodity Codes"].append("RAPESEED")
                hierarchy["RAPESEED"] = []
                hierarchy["Commodity Codes"].append("SOYBEAN")
                hierarchy["SOYBEAN"] = []
                hierarchy["Commodity Codes"].append("SUNSEED")
                hierarchy["SUNSEED"] = []
                hierarchy["Commodity Codes"].append("TUNG")
                hierarchy["TUNG"] = []
            if "CASTOR" in topics[i] and topics[i] != "CASTORSEED":
                hierarchy["CASTORSEED"].append(topics[i])
                leaves.append(topics[i])
            elif  "COCONUT" in topics[i] and topics[i] != "COCONUT" :
                hierarchy["COCONUT"].append(topics[i])
                leaves.append(topics[i])
            elif "CORN" in topics[i] and topics[i] != "CORN" :
                hierarchy["CORN"].append(topics[i])
                leaves.append(topics[i])
            elif "COTTON" in topics[i] and topics[i] != "COTTON":
                hierarchy["COTTON"].append(topics[i])
                leaves.append(topics[i])

            elif "GROUNDNUT" in topics[i] and topics[i] != "GROUNDNUT":
                hierarchy["GROUNDNUT"].append(topics[i])
                leaves.append(topics[i])
            elif "LIN" in topics[i] and topics[i] != "LINSEED":
                hierarchy["LINSEED"].append(topics[i])
                leaves.append(topics[i])
            elif "PALM" in topics[i] and topics[i] != "PALMKERNEL":
                hierarchy["PALMKERNEL"].append(topics[i])
                leaves.append(topics[i])
            elif "RAPE" in topics[i] and topics[i] != "RAPESEED":
                hierarchy["RAPESEED"].append(topics[i])
                leaves.append(topics[i])
            elif "SOY" in topics[i] and topics[i] != "SOYBEAN":
                hierarchy["SOYBEAN"].append(topics[i])
                leaves.append(topics[i])
            elif "SUN" in topics[i] and topics[i] != "SUNSEED":
                hierarchy["SUNSEED"].append(topics[i])
                leaves.append(topics[i])
            elif "TUNG" in topics[i] and topics[i] != "TUNG":
                hierarchy["TUNG"].append(topics[i])
                leaves.append(topics[i])
            else:
                hierarchy["Commodity Codes"].append(topics[i])
                leaves.append(topics[i])
        else:
            if i == 125:
                hierarchy["Energy Codes"] = []
            hierarchy["Energy Codes"].append(topics[i])
            leaves.append(topics[i])
    with open("C:/Users/jcotn/PycharmProjects/BNCL/update/inputs/reuters/hierarchy_file", "wb") as f:
        pickle.dump(hierarchy, f)
    return hierarchy, leaves

def Reuters(data_root):
    topics_vocab = {}
    exchanges_vocab = {}
    orgs_vocab = {}
    people_vocab = {}
    all_places = []
    all_orgs = []
    firsthalf = ""
    isCountry = True
    isfirsthalf = False
    ignorefirst = False
    with open(data_root + '/cat-descriptions_120396.txt') as f:
        section = ""
        for line in f:
            if "_" not in line:
                if line.strip() == "****Subject Codes (135)":
                    section = "topics"
                elif line.strip() == "@heading[Organization Codes (56)]":
                    section = "orgs"
                elif line.strip() == "@heading[Exchange Codes (39)]":
                    section = "exch"
                elif line.strip() == "@heading[Country Codes (176)]":
                    section = "country"
                elif line.strip() == "@heading[People Codes (269)]":
                    section = "people"
                    ignorefirst = True
                elif line.strip() == "International Monetary Fund Managing Director Michel Camdessus (CAMDESSUS)":
                    section = "final"
                elif section == "topics":
                    if "(" in line and "*" not in line:
                        split = line.split("(")
                        long = split[0].strip().upper()
                        short = split[1].split(")")[0].strip().lower()
                        topics_vocab[short] = long
                    elif "(" not in line and "*" not in line and line != "\n":
                        short = line.strip().lower()
                        long = line.strip().upper()
                        topics_vocab[short] = long
                elif section == "orgs" and "@*" in line:
                    split = line.split("(")
                    long = split[0].strip()
                    short = split[1].split(")")[0].strip().lower()
                    orgs_vocab[short] = long
                elif section == "exch" and "@*" in line:
                    split = line.split("(")
                    long = split[0].strip()
                    short = split[1].split(")")[0].strip().lower()
                    exchanges_vocab[short] = long
                elif section == "people" and line != "\n":
                    upper = ""
                    for word in line.split(" "):
                        if word.isupper():
                            upper = word
                    if line.strip() == "@end[itemize]":
                        isCountry = True
                        isfirsthalf = False
                    elif line.strip() == "@begin[itemize]" and not ignorefirst:
                        isCountry = False
                        isfirsthalf = False
                    elif isCountry:
                        country = line.strip()
                        isfirsthalf = False
                        ignorefirst = False
                    elif upper != "" and not isfirsthalf:
                        long = line.split(upper)[0].strip()
                        short = upper.strip("(").split(")")[0].strip().lower()
                        people_vocab[short] = country + " " + long
                        isfirsthalf = False
                    elif upper == "":
                        firsthalf = line.strip()
                        isfirsthalf = True
                    elif isfirsthalf:
                        totalLine = firsthalf + " " + line.strip()
                        long = totalLine.split(upper.strip())[0].strip()
                        short = upper.strip("(").split(")")[0].strip().lower()
                        people_vocab[short] = country + " " + long
                        isfirsthalf = False
                if section == "final" and line != "\n":
                    upper = ""
                    for word in line.split(" "):
                        if word.isupper():
                            upper = word
                    if upper.strip() == "UNCTAD":
                        people_vocab["dadzie"] = "UNCTAD (United Nations Conference on Trade and Development) Secretary-General Kenneth Dadzie"
                    elif upper.strip() == "(DADZIE)":
                        isfirsthalf = False
                    elif upper != "" and not isfirsthalf:
                        long = line.split(upper)[0].strip()
                        short = upper.strip("(").split(")")[0].strip().lower()
                        people_vocab[short] = long
                        isfirsthalf = False
                    elif upper == "":
                        firsthalf = line.strip()
                        isfirsthalf = True
                    elif isfirsthalf:
                        totalLine = firsthalf + " " + line.strip()
                        long = totalLine.split(upper.strip())[0].strip()
                        short = upper.strip("(").split(")")[0].strip().lower()
                        people_vocab[short] = long
                        isfirsthalf = False
    topics_vocab["castorseed"] = "CASTOR SEED"
    topics_vocab["citruspulp"] = "CITRUS PULP"
    topics_vocab["cornglutenfeed"] = "CORN GLUTEN FED"
    topics_vocab["palmkernel"] = "PALM KERNEL"
    topics_vocab["sunseed"] = "SUN SEED"


    with open(data_root + '/all-places-strings.lc.txt') as f:
        for line in f:
            all_places.append(line.strip())

    orgs_vocab["geplacea"] = "Grupo de Paises Latinoamericanos y del Caribe Exportadores de Azucar [Group of Latin American and Caribbean Sugar Exporting Countries]"

    #splits = datasets.SplitGenerator( name=datasets.Split.TRAIN, gen_kwargs={"filepaths": filepaths, "split": "TRAIN", "files": dl_manager.iter_archive(archive)}
    filepaths = ["/reut2-" + "%03d" % i + ".sgm" for i in range(22)]
    trainsamples, trainmetadata = _generate_examples(filepaths, "TRAIN", data_root,  people_vocab, orgs_vocab, topics_vocab, exchanges_vocab)
    testsamples, testmetadata = _generate_examples(filepaths, "TEST", data_root,  people_vocab, orgs_vocab, topics_vocab, exchanges_vocab)
    samples = {"train": trainsamples,
        "test": testsamples}
    metadata = {"train": trainmetadata, "test": testmetadata}
    return {"samples": samples, "metadata": metadata, "all places": all_places, "people_vocab": people_vocab,
            "orgs_vocab": orgs_vocab, "topics_vocab": topics_vocab, "exchanges_vocab": exchanges_vocab}


def _generate_examples(filepaths, split, data_root, people_vocab, orgs_vocab, topics_vocab, exchanges_vocab):
    """This function returns the examples in the raw (text) form."""
    Samples = {}
    Metadata = {}

    for path in filepaths:
        with open(data_root + path, encoding="utf-8", errors="ignore") as f:
            # only the file reut2-017 has one line non UTF-8 encoded so we can ignore it
            line = f.readline()
            while line:
                if line.startswith("<REUTERS"):
                    lewis_split = ""
                    cgis_split = ""
                    old_id = ""
                    new_id = ""
                    topics = []
                    places = []
                    people = []
                    orgs = []
                    exchanges = []
                    date = ""
                    title = ""
                    text = ""
                    text_type = ""
                    line = line.split()
                    lewis_split = line[2].split("=")[1]
                    cgis_split = line[3].split("=")[1]
                    old_id = line[4].split("=")[1]
                    new_id = line[5].split("=")[1][:-1]
                    has_topic = line[1].split("=")[1]
                    line = f.readline()
                    if (
                        (
                            (
                                (split not in lewis_split)
                                or (split == "TRAIN" and has_topic not in ['"YES"', '"NO"'])
                                or (split == "TEST" and has_topic not in ['"YES"', '"NO"'])
                                or (split == "NOT-USED" and has_topic not in ['"YES"', '"NO"', '"BYPASS"'])
                            )
                        )

                    ):  # skip example that are not in the current split
                        li = line
                        while li and not li.startswith("<REUTERS"):
                            li = f.readline()
                        if li:
                            line = li
                elif line.startswith("<TOPICS>"):
                    if line.replace("\n", "") != "<TOPICS></TOPICS>":
                        line = line.split("<D>")
                        topics = [topic.replace("</D>", "") for topic in line[1:]]
                        topics = [topic.replace("</TOPICS>\n", "") for topic in topics]
                    line = f.readline()
                elif line.startswith("<PLACES>"):
                    if line.replace("\n", "") != "<PLACES></PLACES>":
                        line = line.split("<D>")
                        places = [place.replace("</D>", "") for place in line[1:]]
                        places = [place.replace("</PLACES>\n", "") for place in places]
                    line = f.readline()
                elif line.startswith("<PEOPLE>"):
                    if line.replace("\n", "") != "<PEOPLE></PEOPLE>":
                        line = line.split("<D>")
                        people = [p.replace("</D>", "") for p in line[1:]]
                        people = [p.replace("</PEOPLE>\n", "") for p in people]
                    line = f.readline()
                elif line.startswith("<ORGS>"):
                    if line.replace("\n", "") != "<ORGS></ORGS>":
                        line = line.split("<D>")
                        orgs = [org.replace("</D>", "") for org in line[1:]]
                        orgs = [org.replace("</ORGS>\n", "") for org in orgs]
                    line = f.readline()
                elif line.startswith("<EXCHANGES>"):
                    if line.replace("\n", "") != "<EXCHANGES></EXCHANGES>":
                        line = line.split("<D>")
                        exchanges = [ex.replace("</D>", "") for ex in line[1:]]
                        exchanges = [ex.replace("</EXCHANGES>\n", "") for ex in exchanges]
                    line = f.readline()
                elif line.startswith("<DATE>"):
                    date = line.replace("\n", "")
                    date = line[6:-8]
                    line = f.readline()
                elif line.startswith("<TITLE>"):
                    title = line[7:-9]
                    line = f.readline()
                elif "*<TITLE>" in line:
                    # These lines start with a variable number of * chars
                    title = line.split("*<TITLE>")[1][:-1]
                    line = f.readline()
                    while "</TITLE>" not in line:
                        # Convert any \n in TYPE="BRIEF" text to spaces to match other titles
                        title += " " + line[:-1]
                        line = f.readline()
                elif "<BODY>" in line:
                    text = line.split("<BODY>")[1]
                    line = f.readline()
                    while "</BODY>" not in line:
                        text += line
                        line = f.readline()
                elif line.startswith('<TEXT TYPE="UNPROC">'):
                    text_type = '"UNPROC"'
                    text = line[20:]
                    line = f.readline()
                    while "</TEXT>" not in line:
                        text += line
                        line = f.readline()
                elif line.startswith('<TEXT TYPE="BRIEF">'):
                    text_type = '"BRIEF"'
                    line = f.readline()
                elif line.startswith("<TEXT>"):
                    text_type = '"NORM"'
                    line = f.readline()
                elif line.startswith("</REUTERS>"):
                    if topics != []:
                        Samples[title + " " + text] = [topics_vocab[topic] for topic in topics]
                        Metadata[title + " " + text] = {"topics": [topics_vocab[topic] for topic in topics],
                            "lewis_split": lewis_split,
                            "cgis_split": cgis_split,
                            "places": places,
                            "people": [people_vocab[person] for person in people],
                            "orgs": [orgs_vocab[org] for org in orgs],
                            "exchanges": [exchanges_vocab[exchange] for exchange in exchanges],
                            "date": date,
                            "text_type": text_type,
                        }
                    line = f.readline()
                else:
                    line = f.readline()
    return Samples, Metadata