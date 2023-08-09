import streamlit as st
import streamlit.components.v1 as stc
from pathlib import Path
import parser
import fitz
from nltk.tokenize import word_tokenize
import gensim
from gensim.models.phrases import Phraser, Phrases
import image as images
from gensim.models import Word2Vec
import string
import collections
import re
from nltk.corpus import stopwords
import nltk
#nltk.download('all', halt_on_error=False)
nltk.download('stopwords')
nltk.download('punkt')
#nltk.download('wordnet_ic')
import re
import spacy
from spacy.matcher import Matcher
import phonenumbers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# from tika import parserimport spacy
import spacy
from tika import parser
from spacy.matcher import Matcher
import nlp
import spacy
from spacy.matcher import Matcher
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
import PyPDF2
import os
import collections
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
import re
import spacy
from nltk.corpus import stopwords
from matplotlib import pyplot as plt


def extract_skillset(df,text):
	def extract_skills(resume_text):
		nlp = spacy.load('en_core_web_sm')
		nlp_text = nlp(resume_text)

		# removing stop words and implementing word tokenization
		tokens = [token.text for token in nlp_text if not token.is_stop]
		
		skills = ["machine learning",
				"deep learning",
				"nlp",
				"natural language processing",
				"mysql",
				"sql",
				"django",
				"computer vision",
				"tensorflow",
				"opencv",
				"mongodb",
				"artificial intelligence",
				"ai",
				"flask",
				"robotics",
				"data structures",
				"python",
				"c++",
				"matlab",
				"css",
				"html",
				"github",
				"php"]
		
		skillset = []
		
		# check for one-grams (example: python)
		for token in tokens:
			if token.lower() in skills:
				skillset.append(token)
		
		# check for bi-grams and tri-grams (example: machine learning)
		for token in nlp_text.noun_chunks:
			token = token.text.lower().strip()
			if token in skills:
				skillset.append(token)
		
		return [i.capitalize() for i in set([i.lower() for i in skillset])]

	skills = []

	for i in range(0,len(text)): 
		skills.append(extract_skills(text[i]))
	df['Skills'] = skills
	return df

def extract_experience(df,text):
	extracted_text = []
	sub_patterns = ['[A-Z][a-z]* [A-Z][a-z]* Private Limited','[A-Z][a-z]* [A-Z][a-z]* Pvt. Ltd.','[A-Z][a-z]* [A-Z][a-z]* Inc.', '[A-Z][a-z]* LLC',
                '[A-Z][a-z]*[A-Z][a-z]*[A-Z][a-z]* [A-Z][a-z]* & Co.']
	pattern = '({})'.format('|'.join(sub_patterns))
	for i in range(0,len(text)):
		Exp = re.findall(pattern, text[i])
		extracted_text.append(Exp)
	df['Experience'] = extracted_text
	return df

def extract_qual(df,text):
	def extract_education(resume_text):
		nlp = spacy.load('en_core_web_sm')

		# Grad all general stop words
		STOPWORDS = set(stopwords.words('english'))

		# Education Degrees
		EDUCATION = [
					'BE','B.E.', 'B.E', 'BS', 'B.S', 
					'ME', 'M.E', 'M.E.', 'MS', 'M.S', 'MSC','M.Sc.',
					'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 
					'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII',' High School','Secondary SchooL'
				]
		nlp_text = nlp(resume_text)

		# Sentence Tokenizer
		nlp_text = [sent.text.strip() for sent in nlp_text.sents]
		edu = {}
		flag = False
		# Extract education degree
		for index, text in enumerate(nlp_text):        
			for tex in text.split():
				# Replace all special symbols
				tex = re.sub(r'[?|$|.|!|,]', r'', tex)
				# print(tex)
				if tex.upper() in EDUCATION and tex not in STOPWORDS:
					edu[tex] = text + nlp_text[index + 1]
					flag = True
				if tex.upper() == 'HIGH' or tex.upper()=='SECONDARY' :
					if(flag == False):
						edu[tex+' School'] = text + nlp_text[index + 1]
			

		education = []
		for key in edu.keys():
			year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
			if year:
				education.append((key, ''.join(year[0])))
			else:
				education.append(key)
		return education

	qual = []
	for i in range(0,len(text)):
		education = extract_education(text[i])
		qual.append(education)
	qual1 = []
	for tups in qual:
		res = ['-'.join(i) for i in tups]
		qual1.append(res)
	# qual1
	df['Qualifications'] = qual1
	return df


def extract_univ(df,text):
	def extract_university(text,file):
		df = pd.read_csv(file,header=None)
		universities = [i.lower() for i in df[1]]
		college_name = []
		listex = universities
		listsearch = [text.lower()]

		for i in range(len(listex)):
			for ii in range(len(listsearch)):
		#             print(listex[i])
					if re.findall(listex[i], re.sub(' +', ' ', listsearch[ii])):
	#                         print(str(ii)+': '+listex[i])
							college_name.append(listex[i].upper())
	#     print(college_name)
		return college_name
	university = {}
	univ = []
	for i in range(0,len(text)):
		university[i] = extract_university(text[i],'world-universities.csv')
	univ = university.values()
	df['Institutes'] = univ
	return df
			

def extract_contact(df,text):
	def find_phone(text):
		try:
			return list(iter(phonenumbers.PhoneNumberMatcher(text, None)))[0].raw_string
		except:
			return re.search(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', text).group()
    
	list2 = []
	for i in range(0,len(text)):
		phone_number = find_phone(text[i])
		print(phone_number)
		list2.append(phone_number)
	df['Contact'] = list2
	return df

def extract_emailid(text, extracted_text):
	for i in range(0,len(text)):
		r = re.compile(r'[\w\.-]+@[\w\.-]+')
		email = r.findall(text[i])
		extracted_text[i]=email
	text2 = []
	extracted_text2 = {}
	for i in range(0,len(extracted_text)):
		text3 = extracted_text[i][0]
		extracted_text2[i] = text3
		text2.append(text3)
	df = pd.DataFrame(text2)
	df.rename(columns = {0:'Email'}, inplace = True)
	return df	

def extract_name(df,text):  
	list2 = []
	nlp = spacy.load('en_core_web_sm')
	for i in range(0,len(text)):
		if i <1:
			matcher = Matcher(nlp.vocab)
			nlp_text = nlp(text[i])
			pattern1 = [{'POS': 'PROPN'},{'POS': 'PROPN'}]    
			matcher.add('NAME', [pattern1], on_match = None)    
			matches = matcher(nlp_text)    
			for match_id, start, end in matches:
				span = nlp_text[start:end]
				print(span.text)
				list2.append(span.text)
				break
		else:
			matcher = Matcher(nlp.vocab)
			nlp_text = nlp(text[i])
			pattern = [{'POS': 'PROPN'}]
			pattern1 = [{'POS': 'PROPN'},{'POS': 'PROPN'}]    
			matcher.add('NAME', [pattern], on_match = None)
			matches = matcher(nlp_text)    
			for match_id, start, end in matches:
				span = nlp_text[start:end]
				print(span.text)
				list2.append(span.text)
				break
	df['Name'] = list2
	return df

def extract_name1(resume_text):
    
    nlp_text = nlp(resume_text)
    matcher = Matcher(nlp.vocab)
    # print(nlp_text)
    
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}]
    pattern1 = [{'POS': 'PROPN'},{'POS': 'PROPN'}]
    
    matcher.add('NAME', [pattern1,pattern], on_match = None)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text

def pdfextract(file):
    pdf_file = open(file, 'rb')
    read_pdf = PyPDF2.PdfReader(pdf_file)
    number_of_pages = len(read_pdf.pages)
    c = collections.Counter(range(number_of_pages))
    for i in c:
        #page
        page = read_pdf.pages[i]
        page_content = page.extract_text()
    return (page_content.encode('utf-8'))

def create_bigram(words):
    common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    x = words.split()
# Create the relevant phrases from the list of sentences:
    phrases = Phrases(x, common_terms=common_terms)
# The Phraser object is used from now on to transform sentences
    bigram = Phraser(phrases)
# Applying the Phraser to transform our sentences is simply
    all_sentences = list(bigram[x])


def create_profile(file):
	model = Word2Vec.load("final.model")
	file_data = []
	text1 = str(pdfextract(file))
	file_data = fitz.open(file)
	ext1 = ""
	for page in file_data:
		ext1+=page.get_text()    
	content2 = ext1
    # content2 = file_data['content']
	text = text1.replace("\\n", "")
	text = text.lower()
	stats = [nlp(text[0]) for text in model.wv.most_similar("statistics")]
	NLP = [nlp(text[0]) for text in model.wv.most_similar("language")]
	ML = [nlp(text[0]) for text in model.wv.most_similar("machine_learning")]
	DL = [nlp(text[0]) for text in model.wv.most_similar("deep")]
	python = [nlp(text[0]) for text in model.wv.most_similar("python")]
	Data_Engineering = [nlp(text[0]) for text in model.wv.most_similar("data")]
	print("*******************************************")
	matcher = PhraseMatcher(nlp.vocab)
	matcher.add('Stats', None, *stats)
	matcher.add('ML', None, *ML)
	matcher.add('DL', None, *DL)
	matcher.add('Python', None, *python)
	matcher.add('DE', None, *Data_Engineering)
	doc = nlp(text)
	d = []
	matches = matcher(doc)
	for match_id, start, end in matches:
		rule_id = nlp.vocab.strings[match_id]
		span = doc[start : end]
		d.append((rule_id, span.text))
	keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
	print("KEYWORDS")
	df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
	df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
	df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
	df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
	df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
	print("********************DF********************")
	# print(df)
	base = os.path.basename(file)
	filename = os.path.splitext(base)[0]
	name = filename.split('_')
	print(name)
	name2 = name[0]
	name2 = name2.lower()
	name3 = pd.read_csv(StringIO(name2),names = ['Name'])
	name4 = extract_name1(content2)
	name5 = pd.read_csv(StringIO(name4),names = ['Name'])
	dataf = pd.concat([name5['Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
	dataf['Name'].fillna(dataf['Name'].iloc[0], inplace = True)
	print("******************DATAF**************")
    # print(dataf)
	return(dataf)


def plot(candidate_data):
	import tkinter
	import matplotlib
	from PIL import Image
	matplotlib.use('TkAgg')
	plt.rcParams.update({'font.size':30})
	ax = candidate_data.plot.barh(title="Keywords in Resume according to category", legend=False, figsize=(40,20), stacked=True)
	skills = []
	for j in candidate_data.columns:
		for i in candidate_data.index:
			skill = str(j)+": " + str(candidate_data.loc[i][j])
			skills.append(skill)
	patches = ax.patches
	for skill, rect in zip(skills, patches):
		width = rect.get_width()
		if width > 0:
			x = rect.get_x()
			y = rect.get_y()
			height = rect.get_height()
			ax.text(x + width/2., y + height/2., skill, ha='center', va='center')
	# plt.show()
	plt.savefig('score.png')
	image = Image.open('score.png')	
	st.image(image, caption='Scoring')



def screening():
	st.title("Screening and ranking the resume's !!!")
	file = r'D:\Dhiku\sem 8\NLP package\skills.txt'
	with open(r'D:\Dhiku\sem 8\NLP package\skills.txt',encoding="utf8") as f:
		content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]

	#preprocessing
	x=[]
	for line in content:
		tokens=word_tokenize(line)
		tok=[w.lower() for w in tokens]
		table=str.maketrans('','',string.punctuation)
		strpp=[w.translate(table) for w in tok]
		words=[word for word in strpp if word.isalpha()]
		stop_words=set(stopwords.words('english'))
		words=[w for w in words if not w in stop_words]
		x.append(words)
	texts = x
	# texts
	# len(texts)
	content = texts
	common_terms = ["of", "with", "without", "and", "or", "the", "a"]
	x = texts
	# Create the relevant phrases from the list of sentences:
	phrases = Phrases(x,common_terms=common_terms)
	# The Phraser object is used from now on to transform sentences
	bigram = Phraser(phrases)
	all_sentences = list(bigram[x])
	model = gensim.models.Word2Vec(all_sentences,size=5000,min_count=2,workers=4,window=4)
	model.save("final.model")
	wrds = list(model.wv.vocab)
	# print(len(wrds))
	z = model.wv.most_similar("machine_learning")
	mypath = r'D:\Dhiku\sem 8\NLP package\resumedata'
	#Path for the files
	onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
	sim_words=[k[0] for k in model.wv.most_similar("machine_learning")]
	#Code to execute the above functions 
	final_db = pd.DataFrame()
	i = 0
	while i < len(onlyfiles):
		file = onlyfiles[i]
		# print(file)
		dat = create_profile(file)

		final_db = final_db.append(dat)
		i+=1
		#print(final_db)
		#Code to count words under each category and visualize it through MAtplotlib
	final_db2 = final_db['Keyword'].groupby([final_db['Name'], final_db['Subject']]).count().unstack()
	final_db2.reset_index(inplace = True)
	final_db2.fillna(0,inplace=True)
	candidate_data = final_db2.iloc[:,1:]
	candidate_data.index = final_db2['Name']
	#the candidate profile in a csv format
	cand = candidate_data.to_csv('candidate_profile.csv')
	cand_profile = pd.read_csv('candidate_profile.csv',index_col = 0)

	#plot 
	return cand_profile

def parser():
	st.title("Displaying the parsed contents from the Resume")
	file = r'D:\Dhiku\sem 8\NLP package\resumedata'
	onlyfiles = [os.path.join(file, f) for f in os.listdir(file) if os.path.isfile(os.path.join(file, f))]
	text = {}
	j = 0
	for i in onlyfiles:  
		file_data = fitz.open(i)
		text1 = ""
		for page in file_data:
			text1+=page.get_text()    
		text[j]=text1
		j+=1
	# print("next")
	# print(len(text))
	# print(text)
	extracted_text = {}
	df = extract_emailid(text, extracted_text)
	print(df)
	df = extract_name(df,text)
	print(df)
	df = extract_contact(df,text)
	print(df)
	df = extract_skillset(df,text)
	print(df)
	df = extract_qual(df,text)
	print(df)
	df = extract_univ(df,text)
	print(df)
	df = extract_experience(df,text)
	print(df)
	# df.drop(df.iloc[:, 0:1], inplace=True, axis=1)
	return df,True


def main():
	st.title("Resume Uploader")
	st.subheader("Upload the Resume\'s in PDF format")
	files = st.file_uploader("Upload File",accept_multiple_files=True)
	save_folder = r'D:\Dhiku\sem 8\NLP package\resumedata'
	flag1 = False
	for File in files:
		save_path = Path(save_folder, File.name)
		with open(save_path, mode='wb') as w:
			w.write(File.getvalue())
		if save_path.exists():
			st.success(f'File {File.name} is successfully saved!')
			flag1 = True
	flag = False
	st.subheader("Click the button to display the parsed contents!!")
	w4 = st.button("Click me")
	# st.write(w4)
	if w4 and flag1:
		df,flag = parser()
		df.to_csv('file1.csv')
		if(flag == True):	
			df = pd.read_csv("file1.csv")
			st.write(df)  # 

	st.subheader("Click the button to display the scoring of resume!!")
	w5 = st.button("Click me to score")
	if w5:
		st.empty()
		df1 = screening()
		# df1 = pd.read_csv('candidate_profile.csv',index_col = 0)
		plot(df1)
		
if __name__ == "__main__":
    main()



