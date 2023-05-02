# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nlp
import matplotlib.pyplot as plt
import os
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json
from os import listdir
import glob
from scipy import spatial
import spacy
from flask import Flask, render_template, request
import pandas as pd
import docx2txt
from output import *
import nltk
import re
import subprocess  # noqa: S404
app = Flask(__name__)


# %% [markdown]
# Pass a third parameter(flag) as 1 in the match_profile() in order to get your recommendations!

# %%
def cosine_similarity(arr1,arr2):
    ans=1- spatial.distance.cosine(arr1,arr2)
    if(np.isnan(ans)):
        return 0
    else:
        return ans
class job_postings:    
    def __init__(self,link):
        self.df2=pd.read_csv(link)
        self.training_range=int(len(self.df2.loc[:,'uniq_id']))
    def check_threshold(threshold,ele):
        if(ele[0]!=threshold[0][0] and abs(ele[1]-threshold[0][1])<0.03):
            return True
        else:
            return False
    def categorize_jobs(self):
        # #Predefined categories
        #Compare similarities of word embeddings
        nlp=spacy.load('en_core_web_lg')
        job_id=self.df2.loc[:,'uniq_id'].tolist()[:self.training_range]
        job_titles=self.df2.loc[:,'jobtitle'].tolist()[:self.training_range]
        job_descriptions=self.df2.loc[:,'jobdescription'].tolist()[:self.training_range]
        final_cat=pd.DataFrame(index=job_id)
        #categories=['Network Engineer','Application Development','Big Data','Data Analyst','Software Developer','DevOps','Software Testing','Front End','Back End','Full Stack','Web Development','Information Security','Mobile developer','System Administrator','Business Analyst','Manager','Cloud']
        categories=['Network Engineer','Full stack','QA/Test Developer','Enterprise application','DevOps','Mobile Developer','Back End','Database Administrator(DBA)','Front End','Game developer','System Administrator','Data Scientist','Business analyst','Sales professional','Product Manager','Information Security','Software Developer/Java Developer','Web Developer','Cloud Computing']
        for category in categories:
            final_cat[category]=np.nan
        for job_t_d in list(zip(job_id,job_titles,job_descriptions)):
            id_job=job_t_d[0]
            job_i=job_t_d[1]
            job_d=job_t_d[2]
            job_title=nlp(job_i.lower())
            job_description=nlp(job_d.lower())
            match_cat_title=dict()
            match_cat_description=dict()
            for category in categories:
                word=nlp(category.lower())
                match_cat_title[category]=job_title.similarity(word)
                match_cat_description[category]=job_description.similarity(word)
            match_cat_title=sorted(match_cat_title.items(),key=lambda x:x[1],reverse=True)
            match_cat_description=sorted(match_cat_description.items(),key=lambda x:x[1],reverse=True)


            #a represents max
            #if(match_cat_title[0][1]>0.5 or match_cat_description[0][1]>0.5):
            a=match_cat_title[0]
            #print(a)
            match_cat_description=list(filter(lambda x: self.check_threshold(match_cat_title,x),match_cat_description))
            if(len(match_cat_description)!=0):
                print(match_cat_description)
                print(id_job)
                #b=match_cat_description[0]
                final_cat.loc[id_job,a[0]]=1
                match_cat_description.extend([(match_cat_title[0][0],1)])
                sum_proportion=sum([x[1] for x in match_cat_description])
                for ele in match_cat_description:
                    final_cat.loc[id_job,ele[0]]=ele[1]/sum_proportion
            else:
                print(id_job)
                final_cat.loc[id_job,a[0]]=1
        return final_cat
    def clean_skills(self):
        extracted_skills=dict()
        job_skills=np.asarray(self.df2.loc[:,"skills"])
        for i in range(self.training_range):
            #print(i)
            #Method 1: Manual pre-processing
            job_id=self.df2.iloc[i,-1]
            #Method 2:Using NLTK
            tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')
            #print(job_skills[i])
            if(pd.isnull(job_skills[i])):
                continue
            stopwords_list=stopwords.words("english")
            tokens=re.split("|".join([","," and","/"," AND"," or"," OR",";"]),job_skills[i])
            tokens=list(set(tokens))
            extracted_skills[job_id]=[]
            extracted_skills[job_id].extend(tokens)
        return extracted_skills
    def extract_skills(self,extracted_skills):
        df_languages=pd.read_excel('./data/job_profile/languages.xlsx')
        df_frameworks=pd.read_csv("./data/job_profile/frameworks.csv")
        df_database=pd.read_csv("./data/job_profile/database.csv")
        df_os=pd.read_csv("./data/job_profile/operating_systems.csv")
        df_plat=pd.read_csv("./data/job_profile/platforms.csv")
        frameworks=df_frameworks.iloc[:,1].tolist()
        frameworks=[x.lower().strip() for x in frameworks]
        #frameworks=[str(x).split(",")[0] for x in df_frameworks.iloc[:,1]]
        languages=list(df_languages.iloc[:,0])
        languages=[x.lower().strip() for x in languages]
        #frameworks=[x.lower().strip().split('\t')[0] for x in frameworks]
        databases=df_database.iloc[:,0].tolist()
        databases=[x.lower().strip() for x in databases]
        op_systems=df_os.iloc[:,0].tolist()
        op_systems=[x.lower().strip() for x in op_systems]
        platforms=df_plat.iloc[:,1].tolist()
        #print(platforms)
        platforms=[x.lower().strip() for x in platforms]
        #print(frameworks)
        new_extracted=dict()
        for ele in extracted_skills.keys():
            final_lang=''
            final_frame=''
            final_others=''
            final_database=''
            final_plat=''
            final_os=''
            #print(extracted_skills[ele])
            for skill in extracted_skills[ele]:
                skill_base=skill.lower().strip()
                #print(skill_base)
                if(skill_base in languages):
                    if(final_lang==''):
                        final_lang=skill_base
                    else:
                        final_lang=final_lang+","+skill_base
                elif(skill_base in frameworks):
                    if(final_frame==''):
                        final_frame=skill_base
                    else:
                        final_frame=final_frame+","+skill_base
                elif(skill_base in databases):
                    if(final_database==''):
                        final_database=skill_base
                    else:
                        final_database=final_database+","+skill_base
                elif(skill_base in op_systems):
                    if(final_os==''):
                        final_os=skill_base
                    else:
                        final_os=final_os+","+skill_base
                elif(skill_base in platforms):
                    if(final_plat==''):
                        final_plat=skill_base
                    else:
                        final_plat=final_plat+","+skill_base
                else:
                    if(final_others==''):
                        final_others=skill_base
                    else:
                        final_others=final_others+","+skill_base
            new_extracted[ele]=[final_lang,final_frame,final_database,final_os,final_plat,final_others]
        print((list(new_extracted.items()))[:100])
        for ele,describe in list(zip(self.df2.loc[:,'uniq_id'],self.df2.loc[:,'jobdescription'].tolist()))[:self.training_range]:
            doc=nlp(describe)
            final_lang=''
            final_frame=''
            final_others=''
            final_database=''
            final_plat=''
            final_os=''
            for ent in doc.ents:
                word=ent.text
                word=word.lower().strip()
                if(word in languages and word not in final_lang and word not in new_extracted[ele][0].split(",")):
                    if(final_lang==''):
                        final_lang=word
                    else:
                        final_lang=final_lang+","+word
                elif(word in frameworks and word not in final_frame and word not in new_extracted[ele][1].split(",")):
                    if(final_frame==''):
                        final_frame=word
                    else:
                        final_frame=final_frame+","+word
                elif(word in databases and word not in final_database and word not in new_extracted[ele][2].split(",")):
                    if(final_database==''):
                        final_database=word
                    else:
                        final_database=final_database+","+word
                elif(word in op_systems and word not in final_os and word not in new_extracted[ele][3].split(",")):
                    if(final_os==''):
                        final_os=word
                    else:
                        final_os=final_os+","+word
                elif(word in platforms and word not in final_plat and word not in new_extracted[ele][4].split(",")):
                    if(final_plat==''):
                        final_plat=word
                    else:
                        final_plat=final_plat+","+word
                else:
                    if(final_others==''):
                        final_others=word
                    else:
                        final_others=final_others+","+word
            if(final_lang!=''):
                new_extracted[ele][0]+=","+final_lang
            if(final_frame!=''):
                new_extracted[ele][1]+=","+final_frame
            if(final_database!=''):
                new_extracted[ele][2]+=","+final_database
            if(final_os!=''):
                new_extracted[ele][3]+=","+final_os
            if(final_plat!=''):
                new_extracted[ele][4]+=","+final_plat
            if(final_others!=''):
                new_extracted[ele][5]+=","+final_others
            #new_extracted[ele]=[final_lang,final_frame,final_database,final_os,final_plat,final_others]
        extracted_skills_df=pd.DataFrame.from_dict(new_extracted,orient='index',columns=['Language','Framework','Database','OS','Platform','Others'])
        return extracted_skills_df
    def create_job_profile(self,extracted_skills_df,domain_df):
        job_id=extracted_skills_df.index.tolist()
        languages_df=pd.DataFrame(index=job_id)
        platforms_df=pd.DataFrame(index=job_id)
        frameworks_df=pd.DataFrame(index=job_id)
        databases_df=pd.DataFrame(index=job_id)
        
        for job,lang,frame,plat,datab in list(zip(job_id,extracted_skills_df.loc[:,'Language'].tolist(),extracted_skills_df.loc[:,'Framework'].tolist(),extracted_skills_df.loc[:,'Platform'].tolist(),extracted_skills_df.loc[:,'Database'].tolist())):
            #Languages
            l=lang.split(",")
            if(lang!=np.nan or lang!=''):
                for ele in l:
                    if(ele==''):
                        continue
                    if(ele not in languages_df.columns):
                        #languages.append(ele)
                        languages_df[ele]=np.nan
                    languages_df.loc[job,ele]=1
            
            #Frameworks
            l=frame.split(",")
            if(frame!=np.nan or frame!=''):
                for ele in l:
                    if(ele==''):
                        continue
                    if(ele not in frameworks_df.columns):
                        #languages.append(ele)
                        frameworks_df[ele]=np.nan
                    frameworks_df.loc[job,ele]=1

            #Platforms
            l=plat.split(",")
            if(plat!=np.nan or plat!=''):
                for ele in l:
                    if(ele==''):
                        continue
                    if(ele not in platforms_df.columns):
                        #languages.append(ele)
                        platforms_df[ele]=np.nan
                    platforms_df.loc[job,ele]=1
            
            #Databases
            l=datab.split(",")
            if(datab!=np.nan or datab!=''):
                for ele in l:
                    if(ele==''):
                        continue
                    if(ele not in databases_df.columns):
                        #languages.append(ele)
                        databases_df[ele]=np.nan
                    databases_df.loc[job,ele]=1
        languages_df=languages_df.reindex_axis(sorted(languages_df.columns), axis=1)
        frameworks_df=frameworks_df.reindex_axis(sorted(frameworks_df.columns), axis=1)
        platforms_df=platforms_df.reindex_axis(sorted(platforms_df.columns), axis=1)
        databases_df=databases_df.reindex_axis(sorted(databases_df.columns), axis=1)
        domain_df=domain_df.reindex_axis(sorted(domain_df.columns), axis=1)
        
        languages_df.index.name=frameworks_df.index.name=platforms_df.index.name=databases_df.index.name=domain_df.index.name='uniq_id'
        languages_df.to_csv("./data/job_profile/languages_job_profile.csv")
        frameworks_df.to_csv("./data/job_profile/frameworks_job_profile.csv")
        platforms_df.to_csv("./data/job_profile/platforms_job_profile.csv")
        databases_df.to_csv("./data/job_profile/databases_job_profile.csv")
        domain_df.to_csv("./data/job_profile/domain_job_profile.csv")
        print(languages_df.columns)
        
    def clean_common_profile(self,df_user,df_job,flag):
        #Shift .net from languages to frameworks
        if(flag=='Language'):
            print(df_job.columns.tolist())
            #bash and bash/shell
            count=0
            for ele in df_user.loc[:,'bash/shell']:
                if(ele==1.0):
                    df_user.ix[count,'bash']=1.0
                count=count+1
            df_user=df_user.drop('bash/shell',axis=1)
            count=0
            for ele in df_job.loc[:,'bash/shell']:
                if(ele==1.0):
                    df_job.ix[count,'bash']=1.0
                count=count+1
            df_job=df_job.drop('bash/shell',axis=1)

        if(flag=='Framework'):
            print(df_user.columns.tolist())
            count=0
            for ele in df_user.loc[:,'nodejs']:
                if(ele==1.0):
                    df_user.ix[count,'node.js']=1.0
                count=count+1
            df_user=df_user.drop('nodejs',axis=1)
            count=0
            for ele in df_job.loc[:,'nodejs']:
                if(ele==1.0):
                    df_job.ix[count,'node.js']=1.0
                count=count+1
            df_job=df_job.drop('nodejs',axis=1)
            
            count=0
            for ele in df_user.loc[:,'angularjs']:
                if(ele==1.0):
                    df_user.ix[count,'angular']=1.0
                count=count+1
            df_user=df_user.drop('angularjs',axis=1)
            count=0
            for ele in df_job.loc[:,'angularjs']:
                if(ele==1.0):
                    df_job.ix[count,'angular']=1.0
                count=count+1
            df_job=df_job.drop('angularjs',axis=1)
            
        if(flag=='Platform'):
            print(df_user.columns.tolist())
        if(flag=='Database'):
            print(df_user.columns.tolist())
            count=0
            for ele in df_user.loc[:,'microsoft sql server']:
                if(ele==1.0):
                    df_user.ix[count,'sql server']=1.0
                count=count+1
            df_user=df_user.drop('microsoft sql server',axis=1)
            count=0
            for ele in df_job.loc[:,'microsoft sql server']:
                if(ele==1.0):
                    df_job.ix[count,'sql server']=1.0
                count=count+1
            df_job=df_job.drop('microsoft sql server',axis=1)
        return df_user,df_job

    #Input is two dataframes    
    def create_common_profile(self,job_profile_path,user_profile_path,output_path,flag=0):
        if(flag==0):
            #Domain
            userprofile=pd.read_csv(user_profile_path+"DevType.csv",index_col='Respondent')
            jobprofile=pd.read_csv(job_profile_path+"domain_job_profile.csv",index_col='Unnamed: 0')
            print("Read from file")
            print(jobprofile.index)
            #jobprofile=jobprofile.reset_index()
            #userprofile=userprofile.reset_index()
            userprofile.drop('Unnamed: 0', axis=1, inplace=True)
            jobprofile.drop('uniq_id', axis=1, inplace=True)
            jobprofile.index.name='uniq_id'
            print("index 2in domain")
            print(jobprofile.index)
            #print(jobprofile.loc[:,'uniq_id'])
            userprofile.rename(columns={'Product manager':'Product Manager','Back-end developer':'Back End','C-suite executive (CEO, CTO, etc.)':'C-suite executive','Data scientist or machine learning specialist':'Data Scientist','Database administrator':'Database Administrator(DBA)','Mobile developer':'Mobile Developer','Desktop or enterprise applications developer':'Enterprise application','DevOps specialist':'DevOps','Front-end developer':'Front End','Full-stack developer':'Full stack','Marketing or sales professional':'Sales professional','QA or test developer':'QA/Test Developer','System administrator':'System Administrator','Game or graphics developer':'Game developer'},inplace=True)
            jobprofile.rename(columns={'Business analyst':'Data or business analyst'},inplace=True)
            print(userprofile.columns)
            print(jobprofile.columns)
            print("index in domain")
            print(jobprofile.index)
            #Present in userprofile but not in jobprofile
            a=list(set(userprofile.columns)-set(jobprofile.columns))
            print(a)
            for i in a:
                if(i!='Respondent'):
                    jobprofile[i]=0
            b=list(set(jobprofile.columns)-set(userprofile.columns))
            print(b)
            for i in b:
                if(i!='uniq_id'):
                    userprofile[i]=0
            #userprofile=userprofile.set_index('Respondent')
            #jobprofile=jobprofile.set_index('uniq_id')
            userprofile=userprofile[sorted(userprofile.columns.tolist())]
            jobprofile=jobprofile[sorted(jobprofile.columns.tolist())]
            #Exclude 

            print(userprofile.columns==jobprofile.columns)

            print(userprofile.columns)
            print(jobprofile.columns)
            userprofile=userprofile[userprofile.columns.tolist()]
            jobprofile=jobprofile[jobprofile.columns.tolist()]
            userprofile.to_csv(output_path+"domain_user_profile.csv")
            jobprofile.to_csv(output_path+"domain_job_profile.csv")

            #Languages
            df_user=pd.read_csv(user_profile_path+"LanguageWorkedWith.csv",index_col='Respondent')
            df_job=pd.read_csv(job_profile_path+"languages_job_profile.csv",index_col=0)
            df_job.index.name='uniq_id'
            print("index is")
            print(df_job.index)
            print(df_user.columns)
            print(df_job.columns)
            df_user.drop('Unnamed: 0', axis=1, inplace=True)
            #df_job.drop('Unnamed: 0', axis=1, inplace=True)
            df_job.rename(columns={'visual basic .net':'vb.net'},inplace=True)
            df_user.columns=list(map(lambda x:x.lower(),df_user.columns))
            df_job.columns=list(map(lambda x:x.lower(),df_job.columns))
            columns_to_add=[]
            a=list(set(df_user.columns)-(set(df_job.columns)))
            print(a)
            for i in a:
                if(i!='Respondent'):
                    df_job[i]=0        
            b=list(set(df_job.columns)-set(df_user.columns))
            print(b)
            for i in b:
                if(i!='uniq_id'):
                    df_user[i]=0
            print(df_job.index)        
            df_user=df_user[sorted(df_user.columns.tolist())]
            df_job=df_job[sorted(df_job.columns.tolist())]
            #df_user=userprofile.reindex_axis(sorted(df_user.columns), axis=1)
            #df_job=jobprofile.reindex_axis(sorted(df_job.columns), axis=1)
            print("index 2")
            print(df_job.index)
            print(len(set(df_user.columns).intersection(df_job.columns)),len(df_user.columns))
            df_user,df_job=self.clean_common_profile(df_user,df_job,'Language')
            print("language is")
            print(df_job.index[0])
            print(df_job.loc[df_job.index[0],:])
            df_user.to_csv(output_path+"languages_profile_user.csv")
            df_job.to_csv(output_path+"languages_profile_job.csv")

            #Frameworks
            df_user=pd.read_csv(user_profile_path+"FrameworkWorkedWith.csv",index_col='Respondent')
            df_job=pd.read_csv(job_profile_path+"frameworks_job_profile.csv",index_col=0) 
            df_job.index.name='uniq_id'
            print(df_user.columns)
            print(df_job.columns)
            df_user.drop('Unnamed: 0', axis=1, inplace=True)
            #df_job.drop('Unnamed: 0', axis=1, inplace=True)
            #df_job.rename(columns={'visual basic .net':'vb.net'},inplace=True)
            df_user.columns=list(map(lambda x:x.lower(),df_user.columns))
            df_job.columns=list(map(lambda x:x.lower(),df_job.columns))

            a=list(set(df_user.columns)-(set(df_job.columns)))
            print(a)
            for i in a:
                if(i!='Respondent'):
                    df_job[i]=0        
            b=list(set(df_job.columns)-set(df_user.columns))
            print(b)
            for i in b:
                if(i!='uniq_id'):
                    df_user[i]=0
            #userprofile=userprofile.reindex_axis(sorted(userprofile.columns), axis=1)
            #jobprofile=jobprofile.reindex_axis(sorted(jobprofile.columns), axis=1)
            df_user=df_user[sorted(df_user.columns.tolist())]
            df_job=df_job[sorted(df_job.columns.tolist())]

            print(len(set(df_user.columns).intersection(df_job.columns)),len(df_user.columns))
            df_user,df_job=self.clean_common_profile(df_user,df_job,'Framework')   
            df_user.to_csv(output_path+"frameworks_profile_user.csv")
            df_job.to_csv(output_path+"frameworks_profile_job.csv")

            #Platforms
            df_user=pd.read_csv(user_profile_path+"PlatformWorkedWith.csv",index_col='Respondent')
            df_job=pd.read_csv(job_profile_path+"platforms_job_profile.csv",index_col=0) 
            print(df_user.columns)
            df_job.index.name='uniq_id'
            print(df_job.columns)
            df_user.drop('Unnamed: 0', axis=1, inplace=True)
            #df_job.drop('Unnamed: 0', axis=1, inplace=True)
            #df_job.rename(columns={'visual basic .net':'vb.net'},inplace=True)
            df_user.columns=list(map(lambda x:x.lower(),df_user.columns))
            df_job.columns=list(map(lambda x:x.lower(),df_job.columns))

            a=list(set(df_user.columns)-(set(df_job.columns)))
            print(a)
            for i in a:
                if(i!='Respondent'):
                    df_job[i]=0
            b=list(set(df_job.columns)-set(df_user.columns))
            print(b)
            for i in b:
                if(i!='uniq_id'):
                    df_user[i]=0
            df_user=df_user[sorted(df_user.columns.tolist())]
            df_job=df_job[sorted(df_job.columns.tolist())]

            print(len(set(df_user.columns).intersection(df_job.columns)),len(df_user.columns))
            df_user,df_job=self.clean_common_profile(df_user,df_job,'Platform')        
            df_user.to_csv(output_path+"platforms_profile_user.csv")
            df_job.to_csv(output_path+"platforms_profile_job.csv")

            #Databases
            df_user=pd.read_csv(user_profile_path+"DatabaseWorkedWith.csv",index_col='Respondent')
            df_job=pd.read_csv(job_profile_path+"databases_job_profile.csv",index_col=0) 
            df_job.index.name='uniq_id'
            print(df_user.columns)
            print(df_job.columns)
            df_user.drop('Unnamed: 0', axis=1, inplace=True)
            #df_job.drop('Unnamed: 0', axis=1, inplace=True)
            #df_job.rename(columns={'visual basic .net':'vb.net'},inplace=True)
            df_user.columns=list(map(lambda x:x.lower(),df_user.columns))
            df_job.columns=list(map(lambda x:x.lower(),df_job.columns))

            a=list(set(df_user.columns)-(set(df_job.columns)))
            print(a)
            for i in a:
                if(i!='Respondent'):
                    df_job[i]=0
            b=list(set(df_job.columns)-set(df_user.columns))
            print(b)
            for i in b:
                if(i!='uniq_id'):
                    df_user[i]=0
            df_user=df_user[sorted(df_user.columns.tolist())]
            df_job=df_job[sorted(df_job.columns.tolist())]

            print(len(set(df_user.columns).intersection(df_job.columns)),len(df_user.columns))
            df_user,df_job=self.clean_common_profile(df_user,df_job,'Database')        
            df_user.to_csv(output_path+"databases_profile_user.csv")
            df_job.to_csv(output_path+"databases_profile_job.csv")
        #flag indicates that a new user profile
    def match_profile(self,input_path,user_id,skillx,domainx,flag=0):
        #Match a given user_id with all jobs in the database
        
        #Check if user id exists
        df=pd.read_csv(input_path+"domain_user_profile.csv",index_col='Respondent')
        #print(df.columns)
        matches=dict()
        if(flag==0):
            if(user_id in df.index):
                userdomain=df.loc[user_id,:]
                #print(userdomain)
                #If it does, retrieve the user profile from input_path
                df=pd.read_csv(input_path+"languages_profile_user.csv",index_col='Respondent')
                userlanguages=df.loc[user_id,:]

                df=pd.read_csv(input_path+"frameworks_profile_user.csv",index_col='Respondent')
                userframeworks=df.loc[user_id,:]

                df=pd.read_csv(input_path+"platforms_profile_user.csv",index_col='Respondent')
                userplatforms=df.loc[user_id,:]

                df=pd.read_csv(input_path+"databases_profile_user.csv",index_col='Respondent')
                userdatabases=df.loc[user_id,:]

                userdomain=np.asarray(userdomain.fillna(0))
                userlanguages=np.asarray(userlanguages.fillna(0))
                userframeworks=np.asarray(userframeworks.fillna(0))
                userplatforms=np.asarray(userplatforms.fillna(0))
                userdatabases=np.asarray(userdatabases.fillna(0))
                #print(userdomain)
            else:
                print("error! user id not in Dataset")
            #If it doesn't,take user profile as input
        else:
            skills=skillx
            domains=[domainx]
            flag=1
            
            #domains=list(map(lambda x:x.lower(),domains))
            skills=list(map(lambda x:x.lower(),skills))                

            userdomain=pd.DataFrame(columns=df.columns)
            dictionary=dict()
            for domain in domains:
                dictionary[domain]=1.0
            userdomain=userdomain.append(dictionary,ignore_index=True)


            df=pd.read_csv(input_path+"languages_profile_user.csv",index_col='Respondent')
            userlanguages=pd.DataFrame(columns=df.columns)
            dictionary=dict()
            for skill in skills:
                if(skill in df.columns):
                    dictionary[skill]=1.0
            userlanguages=userlanguages.append(dictionary,ignore_index=True)

            df=pd.read_csv(input_path+"frameworks_profile_user.csv",index_col='Respondent')
            userframeworks=pd.DataFrame(columns=df.columns)
            dictionary=dict()
            for skill in skills:
                if(skill in df.columns):
                    dictionary[skill]=1.0
            userframeworks=userframeworks.append(dictionary,ignore_index=True)

            df=pd.read_csv(input_path+"platforms_profile_user.csv",index_col='Respondent')
            userplatforms=pd.DataFrame(columns=df.columns)                
            dictionary=dict()
            for skill in skills:
                if(skill in df.columns):
                    dictionary[skill]=1.0
            userplatforms=userplatforms.append(dictionary,ignore_index=True)

            df=pd.read_csv(input_path+"databases_profile_user.csv",index_col='Respondent')
            userdatabases=pd.DataFrame(columns=df.columns)               
            dictionary=dict()
            for skill in skills:
                if(skill in df.columns):
                    dictionary[skill]=1.0
            userdatabases=userdatabases.append(dictionary,ignore_index=True)
            #print(userdomain)
            userdomain=np.asarray(userdomain.iloc[0,:].fillna(0))
            userlanguages=np.asarray(userlanguages.iloc[0,:].fillna(0))
            userframeworks=np.asarray(userframeworks.iloc[0,:].fillna(0))
            userplatforms=np.asarray(userplatforms.iloc[0,:].fillna(0))
            userdatabases=np.asarray(userdatabases.iloc[0,:].fillna(0))
                
        jobdomain=pd.read_csv(input_path+"domain_job_profile.csv",index_col='uniq_id')
        joblanguages=pd.read_csv(input_path+'languages_profile_job.csv',index_col='uniq_id')
        jobframeworks=pd.read_csv(input_path+'frameworks_profile_job.csv',index_col='uniq_id')
        jobplatforms=pd.read_csv(input_path+'platforms_profile_job.csv',index_col='uniq_id')
        jobdatabases=pd.read_csv(input_path+'databases_profile_job.csv',index_col='uniq_id')
        #print(len(jobdomain.index),len(joblanguages.index))
        for i in jobdomain.index:
            #print(i)
            domain=jobdomain.loc[i,:].fillna(0)
            language=joblanguages.loc[i,:].fillna(0)
            framework=jobframeworks.loc[i,:].fillna(0)
            platform=jobplatforms.loc[i,:].fillna(0)
            database=jobdatabases.loc[i,:].fillna(0)
            job_id=str(i)
            domain=np.asarray(domain)
            language=np.asarray(language)
            framework=np.asarray(framework)
            platform=np.asarray(platform)
            database=np.asarray(database)
            #print(len(domain),len(userdomain))
            score=(0.7*cosine_similarity(domain,userdomain))+(0.3*(cosine_similarity(language,userlanguages)+cosine_similarity(framework,userframeworks)+cosine_similarity(platform,userplatforms)+cosine_similarity(database,userdatabases)))
            matches[job_id]=score
            score=(0.7*cosine_similarity(domain,userdomain))+(0.3*(cosine_similarity(language,userlanguages)+cosine_similarity(framework,userframeworks)+cosine_similarity(platform,userplatforms)+cosine_similarity(database,userdatabases)))
            #Initializing job profiles for later access
            self.job_domain=domain
            self.job_language=language
            self.job_framework=framework
            self.job_platform=platform
            self.job_database=database
            
            self.user_domain=userdomain
            self.user_language=userlanguages
            self.user_framework=userframeworks
            self.user_platform=userplatforms
            self.user_database=userdatabases
        matches=sorted(matches.items(),key=lambda x:x[1],reverse=True)
        
        recommendations=matches[:10]
        #print("recommendations are")
        #print(recommendations)
        rows=pd.DataFrame(columns=self.df2.columns)
        count=0
        for i in recommendations:
            row=self.df2[self.df2['uniq_id']==i[0]]
            #rows[count]=np.asarray(row.values.T.tolist()[0])
            rows=rows.append(row.iloc[0])
            count=count+1
            #print(row)
        return rows
            

obj=job_postings("./data/dice_com-job_us_sample.csv")

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

SKILLS_DB = [
    'docker',
    'kubernetes',
    'python',
    'javascript',
    'word',
    'excel',
    'English',
    'nlp',
    'html',
    'css',
    'java',
    'c++',
    'git',
    'flask',
    'nodejs',
    'ccna',
    'DNS',
    'HTTP',
    'VPN',
    'azure',
    'aws',
    'virtualization',
    'jenkins',
    'chef',
    'ansible','puppet',
    'R','beautifulsoup',
    'android',
    'ios','kotlin','swift','flutter'
]

DOMAIN_DB = [
    'Web Developer',
    'Software Developer/Java Developer',
    'Network Engineer',
    'Cloud Computing',
    'Data Scientist',
    'Mobile Developer'
]

skills = []
 
def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None

def extract_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)
 
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
 
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
 
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
 
    # we create a set to keep the results in.
    found_skills = set()
 
    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)
    
    #print(bigrams_trigrams)  #comment this before production

    for x in DOMAIN_DB:
        if x in bigrams_trigrams:
            domain_des=x
 
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)
 
    return found_skills,domain_des

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    text = extract_text_from_docx(file)
    skills,domain = extract_skills(text)
    
    default_skills = ",".join(skills)
    #print(df.to_html())
    #return df.to_html()
    #df.to_html('templates\\table.html')
    #return default
    skillx = default_skills
    print(skillx)
    domainx = domain
    #Pass a third parameter(flag) as 1 in order to get your recommendations!
    rows=obj.match_profile("./data/",3,skillx,domainx,1)
    #rows.drop(['uniq_id'], axis=1)
    del rows['uniq_id']
    del rows['jobdescription']
    del rows['jobid']
    del rows['shift']
    del rows['site_name']
    rows.rename(columns={'advertiserurl': 'Link_To_Apply', 'employmenttype_jobstatus': 'Job_Type','joblocation_address':'Location', 'jobtitle':'Job_Title','company':'Company','skills':'Skills','postdate':'Update'}, inplace=True)
    # print("type of rows",type(rows))
    rows.describe()
    rows.reset_index(drop=True, inplace=True)
    # rows.index = np.arange(1, len(rows) + 1)
    rows.index = rows.index + 1
    table_html = rows.to_html(border=1, render_links=True, escape=False, col_space=3)

    #return str1 + default + str2  
    return render_template('output.html',output_final = table_html)  


app.run(host="0.0.0.0",port=3333)



