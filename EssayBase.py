#coding=utf8
import os, sys
import math
import numpy as np
from xml.dom import minidom
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def _punctuation_normalizer(word):
    return word
    '''
    pmap = {u'。': u'.', u'，': u',', u'：': u':', u'；': u';', u'！': u'!', u'“': u'"', u'”': u'"', u'？': u'?'}
    if word in pmap:
        return pmap[word]
    else:
        return word
    '''

def _load_dictionary(fname):
    stopwords = set()
    with open(fname,'r',encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

#stopwords = _load_dictionary( 'source/punctuation_utf8.txt')
#connectives = _load_dictionary('resource/connectives')

class Arc:
    def __init__(self, parent, relation):
        self.head = parent
        self.relation = relation


class Token:
    def __init__(self, word_str, pos_tag, ne_tag):
        self.string = word_str
        self.pos_tag = pos_tag
        self.ne_tag = ne_tag
        self.ref = self


class Sentence:
    def __init__(self, global_id, local_id, para_id, tokens, arcs, attributes_dict):
        self.attributes_dict = attributes_dict
        self.global_id = global_id        # the index of the sentence in the essay
        self.local_id = local_id          # the index of the sentence in the paragraph
        self.para_id = para_id            # the index of its paragraph
        self.tokens = tokens              # all words withing this sentence
        self.arcs = arcs

    def get_string(self):
        return " ".join(self.get_words())

    def get_words(self):
        return [token.string for token in self.tokens]

    def get_postags(self):
        return [token.pos_tag for token in self.tokens]

    def get_netags(self):
        return [token.ne_tag for token in self.tokens]


class Paragraph:
    def __init__(self, id, first_sent_id, n_sent,paraType):
        self.id = id                        # the index of the paragraph in the essay
        self.first_sent_id = first_sent_id  # the index of the first sentence within the paragraph
        self.n_sent = n_sent                # the number of sentences in this paragraph
        self.label = ""
        self.paratype = paraType


class Essay:
    def __init__(self):  # option{true: file, false: String}
        self.filename = ""        # filename of a XML file
        self.xml_doc = None
        self.str = ""
        self.title_sent = None
        self.sentences = []
        self.paragraphs = []
        self.attributes = {}
        self.score = 0

    def get_title(self):
        if self.title_sent == None:
            self.title_sent = self.sentences[0]
        return self.title_sent

    def get_sentences(self):
        return self.sentences

    def get_content_sentences(self):
        return self.sentences[self.paragraphs[0].n_sent:]

    def get_paragraphs(self):
        return self.paragraphs

    def parse_file(self, filename):
        self.filename = filename
        doc = minidom.parse(self.filename)
        self.xml_doc = doc
        self.parse_doc(doc)

    def parse_str(self, xmlstr):
        self.str = xmlstr
        doc = minidom.parseString(self.str)
        self.xml_doc = doc
        self.parse_doc(doc)

    def parse_doc(self, doc):
        
        self.score = doc.getElementsByTagName("xml4nlp")[0].getElementsByTagName('doc')[0].getAttribute('type')
        
        root = doc.getElementsByTagName("xml4nlp")[0].getElementsByTagName("doc")[0]
        # print root
        self.attributes = dict([(root.attributes.item(i).name, root.attributes.item(i).value) \
                                for i in range(root.attributes.length)])


        # print self.attributes
        self.essayType = root.getAttribute("essayType")
        # print self.essayType

        para_nodes = root.getElementsByTagName("para")
        # print para_nodes

        if len(para_nodes) > 0:
            global_id = 0
            para_id = 0
            for paraNode in para_nodes:
                para_type = paraNode.getAttribute("type")
                sents = paraNode.getElementsByTagName("sent")
                para_sents = []
                for localId, sent in enumerate(sents):
                    sentence = self._get_sent_object(sent, global_id, localId, para_id)
                    # print sentence.get_string()
                    if sentence != None and len(sentence.tokens) > 0:
                        if len(sentence.tokens) == 1 and sentence.tokens[0] in [u'“', u'”']:
                            continue
                        else:
                            self.sentences.append(sentence)
                            # print sentence.get_string()
                            # print type(sentence)
                            para_sents.append(sentence)
                            global_id += 1

                if len(para_sents) > 0:
                    paraObj = Paragraph(para_id, para_sents[0].global_id, len(para_sents),para_type)
                    self.paragraphs.append(paraObj)
                    para_id += 1


    def _get_sent_object(self, sentNode, globalId, localId, paraId):
        attributes_dict = dict(
            [(sentNode.attributes.item(i).name, sentNode.attributes.item(i).value) \
             for i in range(sentNode.attributes.length)]
            )
        # print attributes_dict
        tokens = []
        arcs = []

        wordNodes = sentNode.getElementsByTagName("word")
        #print 'Sentence has word:', len(wordNodes)
        for wordNode in wordNodes:
            word = wordNode.getAttribute("cont")
            word = _punctuation_normalizer(word)
            pos = wordNode.getAttribute("pos")
            entity = wordNode.getAttribute("ne")
            token = Token(word, pos, entity)
            tokens.append(token)
            arcs.append(Arc(int(wordNode.getAttribute("parent")),
                            wordNode.getAttribute("relate")))

        sent = Sentence(globalId, localId, paraId, tokens, arcs, attributes_dict)
        # print sent.get_string()
        sent.label = sentNode.getAttribute("type")

        if sent.label == 'noisySen':
            print('noisySen')
            return None

        segmentend = sentNode.getAttribute("segmentend").lower()
        if segmentend == 'end':
            sent.segment_end = True
        return sent

    def printEssayInfo(self):
        for sent in self.sentences:
            print (sent.get_string())

    def to_xml(self):
        """ Convert to ltp xml(string) """
        # xml object
        et_root = ET.Element('xml4nlp')
        et_doc = ET.Element('doc', self.attributes)
        et_root.append(et_doc)
        # Add paragraph/sent/word elements to doc and root
        for para_id, para in enumerate(self.paragraphs):
            et_para = ET.Element('para', {'id':str(para_id+1), 'type': para.paratype})
            et_doc.append(et_para)

            for sent_id in range(para.first_sent_id, para.first_sent_id+para.n_sent):
                sent = self.sentences[sent_id]
                sent.attributes_dict['cont'] = ''.join([token.string for token in sent.tokens])
                attrib_dict = sent.attributes_dict
                #attrib_dict['type'] = '' # For tag clean
                et_sent = ET.Element('sent', attrib_dict)

                for token_id, token in enumerate(sent.tokens):
                    arc = sent.arcs[token_id]
                    postag = token.pos_tag
                    netag = token.ne_tag
                    et_word = ET.Element('word',
                                {'id':str(token_id), 'cont':token.string, 'ne':netag,
                                'parent':str(arc.head),
                                'relate':arc.relation, 'pos':postag})
                    et_sent.append(et_word)

                et_para.append(et_sent)

        docxml = minidom.parseString(ET.tostring(et_root, encoding='UTF-8'))
        return docxml.toprettyxml(encoding='UTF-8')


def get_filenames(rootDir):
    names = []
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        
        if os.path.isdir(path): 
            names.extend(os.path.convert(path))
        else:
            names.append(path)
            
    return names

def tag_clean(dirname, newdir):
    filenames = os.listdir(dirname)

    for fname in filenames:
        essay = Essay()
        essay.parse_file(os.path.join(dirname, fname))
        xmlstring = essay.to_xml()
        with open(os.path.join(newdir, fname), 'w') as fout:
            print >> fout, xmlstring

def getthesis():
    newdir = 'LELE-Annotated-Clean'
    filenames = os.listdir(newdir)
    w_file = 'source/thesis.data'
    w = open(w_file,'w',encoding='utf-8')
    for file in filenames:
        fname = os.path.join(newdir, file)
        essay = Essay()
        essay.parse_file(fname)

        for sent in essay.sentences:
            if sent.label in ['thesisSen','ideaSen','ideaendSen','conclusionSen'] :
                thesis = [token.string for token in sent.tokens]
                pos = [token.pos_tag for token in sent.tokens]
                w.write(fname+'\t'+sent.label+'\t'+' '.join(thesis)+'\n')
                w.write(' '.join(pos)+'\n')
    w.close()





if __name__ == "__main__":
   getthesis()



