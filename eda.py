import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#import spacy
import re
#import emoji

# For printing option and text color
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


df_sub = pd.read_csv('Kaggle_resources/datasets/sample_submission.csv')
df_test = pd.read_csv('Kaggle_resources/datasets/test.csv')
test_df = df_test.set_index('tweetid')
df_train = pd.read_csv('Kaggle_resources/datasets/train.csv')
train_df = df_train.set_index('tweetid')



#Exploratory Data Analysis
def explore_data():
    def view_data():
        return (train_df.head(), train_df.shape, test_df.head(), test_df.shape)

    def null_value_check():
        print(f'No. of empty messages in train: {len(blanks_train)}\n')
        print(f'No. of empty messages in test: {len(blanks_test)}')

    def sentiment_distribution():
        # Count of classes in sentiment 
        sns.set(style="darkgrid",palette='summer')
        ax = sns.countplot(x='sentiment', data=train_df)

    def class_dist_perct():
        print(color.BOLD +'Percentage of a particular `Class` in the train dataset\n'+ color.END)
        print(f'Class 2 ~ News \n{round((df_train.sentiment.value_counts()[2]/len(df_train))*100,2)} %\n')
        print(f'Class 1 ~ Pro \n{round((df_train.sentiment.value_counts()[1]/len(df_train))*100,2)} %\n')
        print(f'Class 0 ~ Neutral \n{round((df_train.sentiment.value_counts()[0]/len(df_train))*100,2)} %\n')
        print(f'Class -1 ~ Anti \n{round((df_train.sentiment.value_counts()[-1]/len(df_train))*100,2)} %')




#Preprocessing
def preprocess():
# checking and removing duplicates
    def duplicate_remover(df,column_name):
        copy = df.copy()
        cn = column_name
        i = 0
        for tweet in copy[cn]:
            
            if i in copy.index:
                
                if (copy[cn]==copy[cn][i]).sum() > 1:
                    dup_index = list(copy[copy[cn]==copy[cn][i]].index)
                    dup_index.pop(0)
                    copy.drop(axis=0,index=dup_index,inplace=True)
                    copy.reset_index(drop=True)
                i=i+1
            else:
                i=i+1
        return copy.reset_index(drop=True)

    train = duplicate_remover(train_df,'message')
    train = duplicate_remover(train_df,'message')
    print(train.shape)
    print(test.shape)

    train_diff = train_df.sentiment.value_counts() - train.sentiment.value_counts()
    train_diff

    # Finds URL's in a string
    def find_url(string): 
        text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)
        return "".join(text)


    train['url'] = train.message.apply(lambda x:find_url(x))
    test['url']= test.message.apply(lambda x:find_url(x))




    # Finds and removes url in tweets boby 
    train['message']=train['message'].apply(lambda x: re.sub(
        pattern='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        repl='',
        string=x))

    test['message']=test['message'].apply(lambda x: re.sub(
        pattern='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        repl='',
        string=x))



    # Corrects/replaces contractions with full words

    def correct_contraction(tweet):
        tweet = re.sub(r"he's", "he is", tweet)
        tweet = re.sub(r"there's", "there is", tweet)
        tweet = re.sub(r"We're", "We are", tweet)
        tweet = re.sub(r"That's", "That is", tweet)
        tweet = re.sub(r"won't", "will not", tweet)
        tweet = re.sub(r"they're", "they are", tweet)
        tweet = re.sub(r"Can't", "Cannot", tweet)
        tweet = re.sub(r"wasn't", "was not", tweet)
        tweet = re.sub(r"aren't", "are not", tweet)
        tweet = re.sub(r"isn't", "is not", tweet)
        tweet = re.sub(r"What's", "What is", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"should've", "should have", tweet)
        tweet = re.sub(r"where's", "where is", tweet)
        tweet = re.sub(r"we'd", "we would", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"weren't", "were not", tweet)
        tweet = re.sub(r"They're", "They are", tweet)
        tweet = re.sub(r"let's", "let us", tweet)
        tweet = re.sub(r"it's", "it is", tweet)
        tweet = re.sub(r"can't", "cannot", tweet)
        tweet = re.sub(r"don't", "do not", tweet)
        tweet = re.sub(r"you're", "you are", tweet)
        tweet = re.sub(r"i've", "I have", tweet)
        tweet = re.sub(r"that's", "that is", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"doesn't", "does not", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"didn't", "did not", tweet)
        tweet = re.sub(r"ain't", "am not", tweet)
        tweet = re.sub(r"you'll", "you will", tweet)
        tweet = re.sub(r"I've", "I have", tweet)
        tweet = re.sub(r"Don't", "do not", tweet)
        tweet = re.sub(r"I'll", "I will", tweet)
        tweet = re.sub(r"I'd", "I would", tweet)
        tweet = re.sub(r"Let's", "Let us", tweet)
        tweet = re.sub(r"you'd", "You would", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"Ain't", "am not", tweet)
        tweet = re.sub(r"Haven't", "Have not", tweet)
        tweet = re.sub(r"Could've", "Could have", tweet)
        tweet = re.sub(r"youve", "you have", tweet)
        tweet = re.sub(r"haven't", "have not", tweet)
        tweet = re.sub(r"hasn't", "has not", tweet)
        tweet = re.sub(r"There's", "There is", tweet)
        tweet = re.sub(r"He's", "He is", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"You're", "You are", tweet)
        tweet = re.sub(r"I'M", "I am", tweet)
        tweet = re.sub(r"shouldn't", "should not", tweet)
        tweet = re.sub(r"wouldn't", "would not", tweet)
        tweet = re.sub(r"i'm", "I am", tweet)
        tweet = re.sub(r"I'm", "I am", tweet)
        tweet = re.sub(r"Isn't", "is not", tweet)
        tweet = re.sub(r"Here's", "Here is", tweet)
        tweet = re.sub(r"you've", "you have", tweet)
        tweet = re.sub(r"we're", "we are", tweet)
        tweet = re.sub(r"what's", "what is", tweet)
        tweet = re.sub(r"couldn't", "could not", tweet)
        tweet = re.sub(r"we've", "we have", tweet)
        tweet = re.sub(r"who's", "who is", tweet)
        tweet = re.sub(r"y'all", "you all", tweet)
        tweet = re.sub(r"would've", "would have", tweet)
        tweet = re.sub(r"it'll", "it will", tweet)
        tweet = re.sub(r"we'll", "we will", tweet)
        tweet = re.sub(r"We've", "We have", tweet)
        tweet = re.sub(r"he'll", "he will", tweet)
        tweet = re.sub(r"Y'all", "You all", tweet)
        tweet = re.sub(r"Weren't", "Were not", tweet)
        tweet = re.sub(r"Didn't", "Did not", tweet)
        tweet = re.sub(r"they'll", "they will", tweet)
        tweet = re.sub(r"they'd", "they would", tweet)
        tweet = re.sub(r"DON'T", "DO NOT", tweet)
        tweet = re.sub(r"they've", "they have", tweet)
        
        #correct some acronyms while we are at it
        tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)
        tweet = re.sub(r"azwx", "Arizona Weather", tweet)  
        tweet = re.sub(r"alwx", "Alabama Weather", tweet)
        tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)      
        tweet = re.sub(r"gawx", "Georgia Weather", tweet)  
        tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  
        tweet = re.sub(r"cawx", "California Weather", tweet)
        tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet) 
        tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)
        tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)
        tweet = re.sub(r"arwx", "Arkansas Weather", tweet)  
        tweet = re.sub(r"lmao", "laughing my ass off", tweet)  
        tweet = re.sub(r"amirite", "am I right", tweet)
        
        #and some typos/abbreviations
        tweet = re.sub(r"w/e", "whatever", tweet)
        tweet = re.sub(r"w/", "with", tweet)
        tweet = re.sub(r"USAgov", "USA government", tweet)
        tweet = re.sub(r"recentlu", "recently", tweet)
        tweet = re.sub(r"Ph0tos", "Photos", tweet)
        tweet = re.sub(r"exp0sed", "exposed", tweet)
        tweet = re.sub(r"<3", "love", tweet)
        tweet = re.sub(r"amageddon", "armageddon", tweet)
        tweet = re.sub(r"Trfc", "Traffic", tweet)
        tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
        tweet = re.sub(r"16yr", "16 year", tweet)
        tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)
        
        #hashtags and usernames
        tweet = re.sub(r"IranDeal", "Iran Deal", tweet)
        tweet = re.sub(r"ArianaGrande", "Ariana Grande", tweet)
        tweet = re.sub(r"camilacabello97", "camila cabello", tweet) 
        tweet = re.sub(r"RondaRousey", "Ronda Rousey", tweet)     
        tweet = re.sub(r"MTVHottest", "MTV Hottest", tweet)
        tweet = re.sub(r"TrapMusic", "Trap Music", tweet)
        tweet = re.sub(r"ProphetMuhammad", "Prophet Muhammad", tweet)
        tweet = re.sub(r"PantherAttack", "Panther Attack", tweet)
        tweet = re.sub(r"StrategicPatience", "Strategic Patience", tweet)
        tweet = re.sub(r"socialnews", "social news", tweet)
        tweet = re.sub(r"IDPs:", "Internally Displaced People :", tweet)
        tweet = re.sub(r"ArtistsUnited", "Artists United", tweet)
        tweet = re.sub(r"ClaytonBryant", "Clayton Bryant", tweet)
        tweet = re.sub(r"jimmyfallon", "jimmy fallon", tweet)
        tweet = re.sub(r"justinbieber", "justin bieber", tweet)  
        tweet = re.sub(r"UTC2015", "UTC 2015", tweet)
        tweet = re.sub(r"Time2015", "Time 2015", tweet)
        tweet = re.sub(r"djicemoon", "dj icemoon", tweet)
        tweet = re.sub(r"LivingSafely", "Living Safely", tweet)
        tweet = re.sub(r"FIFA16", "Fifa 2016", tweet)
        tweet = re.sub(r"thisiswhywecanthavenicethings", "this is why we cannot have nice things", tweet)
        tweet = re.sub(r"bbcnews", "bbc news", tweet)
        tweet = re.sub(r"UndergroundRailraod", "Underground Railraod", tweet)
        tweet = re.sub(r"c4news", "c4 news", tweet)
        tweet = re.sub(r"OBLITERATION", "obliteration", tweet)
        tweet = re.sub(r"MUDSLIDE", "mudslide", tweet)
        tweet = re.sub(r"NoSurrender", "No Surrender", tweet)
        tweet = re.sub(r"NotExplained", "Not Explained", tweet)
        tweet = re.sub(r"greatbritishbakeoff", "great british bake off", tweet)
        tweet = re.sub(r"LondonFire", "London Fire", tweet)
        tweet = re.sub(r"KOTAWeather", "KOTA Weather", tweet)
        tweet = re.sub(r"LuchaUnderground", "Lucha Underground", tweet)
        tweet = re.sub(r"KOIN6News", "KOIN 6 News", tweet)
        tweet = re.sub(r"LiveOnK2", "Live On K2", tweet)
        tweet = re.sub(r"9NewsGoldCoast", "9 News Gold Coast", tweet)
        tweet = re.sub(r"nikeplus", "nike plus", tweet)
        tweet = re.sub(r"david_cameron", "David Cameron", tweet)
        tweet = re.sub(r"peterjukes", "Peter Jukes", tweet)
        tweet = re.sub(r"MikeParrActor", "Michael Parr", tweet)
        tweet = re.sub(r"4PlayThursdays", "Foreplay Thursdays", tweet)
        tweet = re.sub(r"TGF2015", "Tontitown Grape Festival", tweet)
        tweet = re.sub(r"realmandyrain", "Mandy Rain", tweet)
        tweet = re.sub(r"GraysonDolan", "Grayson Dolan", tweet)
        tweet = re.sub(r"ApolloBrown", "Apollo Brown", tweet)
        tweet = re.sub(r"saddlebrooke", "Saddlebrooke", tweet)
        tweet = re.sub(r"TontitownGrape", "Tontitown Grape", tweet)
        tweet = re.sub(r"AbbsWinston", "Abbs Winston", tweet)
        tweet = re.sub(r"ShaunKing", "Shaun King", tweet)
        tweet = re.sub(r"MeekMill", "Meek Mill", tweet)
        tweet = re.sub(r"TornadoGiveaway", "Tornado Giveaway", tweet)
        tweet = re.sub(r"GRupdates", "GR updates", tweet)
        tweet = re.sub(r"SouthDowns", "South Downs", tweet)
        tweet = re.sub(r"braininjury", "brain injury", tweet)
        tweet = re.sub(r"auspol", "Australian politics", tweet)
        tweet = re.sub(r"PlannedParenthood", "Planned Parenthood", tweet)
        tweet = re.sub(r"calgaryweather", "Calgary Weather", tweet)
        tweet = re.sub(r"weallheartonedirection", "we all heart one direction", tweet)
        tweet = re.sub(r"edsheeran", "Ed Sheeran", tweet)
        tweet = re.sub(r"TrueHeroes", "True Heroes", tweet)
        tweet = re.sub(r"ComplexMag", "Complex Magazine", tweet)
        tweet = re.sub(r"TheAdvocateMag", "The Advocate Magazine", tweet)
        tweet = re.sub(r"CityofCalgary", "City of Calgary", tweet)
        tweet = re.sub(r"EbolaOutbreak", "Ebola Outbreak", tweet)
        tweet = re.sub(r"SummerFate", "Summer Fate", tweet)
        tweet = re.sub(r"RAmag", "Royal Academy Magazine", tweet)
        tweet = re.sub(r"offers2go", "offers to go", tweet)
        tweet = re.sub(r"ModiMinistry", "Modi Ministry", tweet)
        tweet = re.sub(r"TAXIWAYS", "taxi ways", tweet)
        tweet = re.sub(r"Calum5SOS", "Calum Hood", tweet)
        tweet = re.sub(r"JamesMelville", "James Melville", tweet)
        tweet = re.sub(r"JamaicaObserver", "Jamaica Observer", tweet)
        tweet = re.sub(r"TweetLikeItsSeptember11th2001", "Tweet like it is september 11th 2001", tweet)
        tweet = re.sub(r"cbplawyers", "cbp lawyers", tweet)
        tweet = re.sub(r"fewmoretweets", "few more tweets", tweet)
        tweet = re.sub(r"BlackLivesMatter", "Black Lives Matter", tweet)
        tweet = re.sub(r"NASAHurricane", "NASA Hurricane", tweet)
        tweet = re.sub(r"onlinecommunities", "online communities", tweet)
        tweet = re.sub(r"humanconsumption", "human consumption", tweet)
        tweet = re.sub(r"Typhoon-Devastated", "Typhoon Devastated", tweet)
        tweet = re.sub(r"Meat-Loving", "Meat Loving", tweet)
        tweet = re.sub(r"facialabuse", "facial abuse", tweet)
        tweet = re.sub(r"LakeCounty", "Lake County", tweet)
        tweet = re.sub(r"BeingAuthor", "Being Author", tweet)
        tweet = re.sub(r"withheavenly", "with heavenly", tweet)
        tweet = re.sub(r"thankU", "thank you", tweet)
        tweet = re.sub(r"iTunesMusic", "iTunes Music", tweet)
        tweet = re.sub(r"OffensiveContent", "Offensive Content", tweet)
        tweet = re.sub(r"WorstSummerJob", "Worst Summer Job", tweet)
        tweet = re.sub(r"HarryBeCareful", "Harry Be Careful", tweet)
        tweet = re.sub(r"NASASolarSystem", "NASA Solar System", tweet)
        tweet = re.sub(r"animalrescue", "animal rescue", tweet)
        tweet = re.sub(r"KurtSchlichter", "Kurt Schlichter", tweet)
        tweet = re.sub(r"aRmageddon", "armageddon", tweet)
        tweet = re.sub(r"Throwingknifes", "Throwing knives", tweet)
        tweet = re.sub(r"GodsLove", "God's Love", tweet)
        tweet = re.sub(r"bookboost", "book boost", tweet)
        tweet = re.sub(r"ibooklove", "I book love", tweet)
        tweet = re.sub(r"NestleIndia", "Nestle India", tweet)
        tweet = re.sub(r"realDonaldTrump", "Donald Trump", tweet)
        tweet = re.sub(r"DavidVonderhaar", "David Vonderhaar", tweet)
        tweet = re.sub(r"CecilTheLion", "Cecil The Lion", tweet)
        tweet = re.sub(r"weathernetwork", "weather network", tweet)
        tweet = re.sub(r"withBioterrorism&use", "with Bioterrorism & use", tweet)
        tweet = re.sub(r"Hostage&2", "Hostage & 2", tweet)
        tweet = re.sub(r"GOPDebate", "GOP Debate", tweet)
        tweet = re.sub(r"RickPerry", "Rick Perry", tweet)
        tweet = re.sub(r"frontpage", "front page", tweet)
        tweet = re.sub(r"NewsInTweets", "News In Tweets", tweet)
        tweet = re.sub(r"ViralSpell", "Viral Spell", tweet)
        tweet = re.sub(r"til_now", "until now", tweet)
        tweet = re.sub(r"volcanoinRussia", "volcano in Russia", tweet)
        tweet = re.sub(r"ZippedNews", "Zipped News", tweet)
        tweet = re.sub(r"MicheleBachman", "Michele Bachman", tweet)
        tweet = re.sub(r"53inch", "53 inch", tweet)
        tweet = re.sub(r"KerrickTrial", "Kerrick Trial", tweet)
        tweet = re.sub(r"abstorm", "Alberta Storm", tweet)
        tweet = re.sub(r"Beyhive", "Beyonce hive", tweet)
        tweet = re.sub(r"IDFire", "Idaho Fire", tweet)
        tweet = re.sub(r"DETECTADO", "Detected", tweet)
        tweet = re.sub(r"RockyFire", "Rocky Fire", tweet)
        tweet = re.sub(r"Listen/Buy", "Listen / Buy", tweet)
        tweet = re.sub(r"yycstorm", "Calgary Storm", tweet)
        tweet = re.sub(r"IDPs:", "Internally Displaced People :", tweet)
        tweet = re.sub(r"ArtistsUnited", "Artists United", tweet)
        tweet = re.sub(r"ENGvAUS", "England vs Australia", tweet)
        tweet = re.sub(r"ScottWalker", "Scott Walker", tweet)

        
        return tweet



    train['message']=train['message'].apply(lambda x: correct_contraction(x))
    test['message']=test['message'].apply(lambda x: correct_contraction(x))


    # Finds @ mentions in tweets
    def find_at(text):
        line=re.findall(r'(?<=@)\w+',text)
        return " ".join(line)


    train['mention']=train['message'].apply(lambda x: find_at(x))
    train['message']=train['message'].apply(lambda x: re.sub(r'(?<=@)\w+','',x))

    test['mention']=test['message'].apply(lambda x: find_at(x))
    test['message']=test['message'].apply(lambda x: re.sub(r'(?<=@)\w+','',x))


    # Finds hashtags in tweet(#)
    def find_hash(text):
        line=re.findall(r'(?<=#)\w+',text)
        return " ".join(line)

    train['hashtags']=train['message'].apply(lambda x: find_hash(x))
    train['message']=train['message'].apply(lambda x: re.sub(r'(?<=#)\w+','',x))

    test['hashtags']=test['message'].apply(lambda x: find_hash(x))
    test['message']=test['message'].apply(lambda x: re.sub(r'(?<=#)\w+','',x))

    # Finds emoji's in a tweet
    def find_emoji(text):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        emo = re.findall(emoji_pattern, text)
        return ''.join(emo)



    train['emoji']=train['message'].apply(lambda x: find_emoji(x))

    test['emoji']=test['message'].apply(lambda x: find_emoji(x))


    train['message']=train['message'].apply(lambda x: re.sub(find_emoji(x),'',x))

    test['message']=test['message'].apply(lambda x: re.sub(find_emoji(x),'',x))

    train.head()

    test.head()