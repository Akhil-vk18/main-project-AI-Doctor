import random
import json
import nltk
nltk.download('punkt_tab')
import torch
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "AI Doctor"
symptom=[]
column_names = ['itching','skin rash','nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
 'vomiting', 'burning micturition',
 'spotting urination', 'fatigue','weight gain','anxiety','cold hands and feets','mood swings','weight loss','restlessness','lethargy','patches in throat','irregular sugar level',
 'cough', 'high fever',
 'sunken eyes', 'breathlessness',
 'sweating', 'dehydration',
 'indigestion', 'headache',
 'yellowish skin', 'dark urine',
 'nausea', 'loss of appetite',
 'pain behind the eyes', 'back pain',
 'constipation', 'abdominal pain',
 'diarrhoea', 'fever',
 'yellow urine', 'yellowing of eyes',
 'acute liver failure', 'fluid overload',
 'swelling of stomach', 'swelled lymph nodes',
 'malaise', 'blurred and distorted vision',
 'phlegm', 'throat irritation',
 'redness of eyes', 'sinus pressure',
 'runny nose', 'congestion',
 'chest pain', 'weakness in limbs',
 'fast heart rate', 'pain during bowel movements',
 'pain in anal region', 'bloody stool',
 'irritation in anus',
 'neck pain',
 'dizziness',
 'cramps',
 'bruising',
 'obesity',
 'swollen legs',
 'swollen blood vessels',
 'puffy face and eyes',
 'enlarged thyroid',
 'brittle nails',
 'swollen extremeties',
 'excessive hunger',
 'extra marital contacts',
 'drying and tingling lips',
 'slurred speech',
 'knee pain',
 'hip joint pain',
 'muscle weakness',
 'stiff neck',
 'swelling joints',
 'movement stiffness',
 'spinning movements',
 'loss of balance',
 'unsteadiness',
 'weakness of one body side',
 'loss of smell',
 'bladder discomfort',
 'foul smell of urine',
 'continuous feel of urine',
 'passage of gases',
 'internal itching',
 'toxic look (typhos)',
 'depression',
 'irritability',
 'muscle pain',
 'altered sensorium',
 'red spots over body',
 'belly pain',
 'abnormal menstruation',
 'dischromic patches',
 'watering from eyes',
 'increased appetite',
 'polyuria',
 'family history',
 'mucoid sputum',
 'rusty sputum',
 'lack of concentration',
 'visual disturbances',
 'receiving blood transfusion',
 'receiving unsterile injections',
 'coma',
 'stomach bleeding',
 'distention of abdomen',
 'history of alcohol consumption',
 'fluid overload.1',
 'blood in sputum',
 'prominent veins on calf',
 'palpitations',
 'painful walking',
 'pus filled pimples',
 'blackheads',
 'scurring',
 'skin peeling',
 'silver like dusting',
 'small dents in nails',
 'inflammatory nails',
 'blister',
 'red sore around nose',
 'yellow crust ooze']
df = pd.DataFrame(columns=column_names)
df.loc[0] = [0] * len(column_names)
disease_model = load_model('pred_model.h5',compile=False)
disease_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
disease=''
disease_names=['(vertigo) Paroymsal  Positional Vertigo','AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis',
 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection',
 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ',
 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis',
 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']
dict={'Fungal infection':'A fungal infection, also called mycosis, is a skin disease caused by a fungus. There are '
                         'millions of species of fungi. They live in the dirt, on plants, on household surfaces, '
                         'and on your skin. Sometimes, they can lead to skin problems like rashes or bumps.',
      'Allergy':'Allergies, also known asallergic diseases, are a number of conditions caused byhypersensitivityÂ '
                'of the a immune systemÂ to typically harmless substances in the environment.Â These diseases '
                'includeÂ hay fever,Â food allergies,Â atopic dermatitis,Â allergic asthma, andÂ anaphylaxis.Â '
                'Symptoms may includeÂ red eyes, an itchy rash,Â sneezing, aÂ runny nose,Â shortness of breath, '
                'or swelling.Â Food intolerancesÂ andÂ food poisoningÂ are separate conditions.',
      'GERD':'Gastroesophageal reflux disease(GERD), is achroniccondition in which stomach contents rise up '
             'into theesophagus, resulting in either symptoms or complications.Symptoms include the taste of acid '
             'in the back of the mouth,heartburn,bad breath,chest pain, regurgitation, breathing problems, '
             'and wearing away of theteeth.Complications includeesophagitis,esophageal stricture, andBarretts '
             'esophagus.',
      'Chronic cholestasis':'Cholecystitisisinflammationof thegallbladder.Symptoms includeright '
                            'upperabdominal pain, nausea, vomiting, and occasionally fever.Oftengallbladder '
                            'attacks(biliary colic) precede acute cholecystitis.The pain lasts longer in '
                            'cholecystitis than in a typical gallbladder attack.Without appropriate treatment, '
                            'recurrent episodes of cholecystitis are common.Complications of acute cholecystitis '
                            'includegallstone pancreatitis,common bile duct stones, orinflammation of the '
                            'common bile duct.',
      'Drug Reaction':'Anadverse drug reaction(ADR) is an injury caused by takingmedication.ADRs may occur '
                      'following a single dose or prolonged administration of adrugor result from the combination '
                      'of two or more drugs. The meaning of this term differs from the term "side effect" because '
                      'side effects can be beneficial as well as detrimental.The study of ADRs is the concern of '
                      'the field known aspharmacovigilance. Anadverse drug event(ADE) refers to any injury '
                      'occurring at the time a drug is used, whether or not it is identified as a cause of the '
                      'injury.An ADR is a special type of ADE in which a causative relationship can be shown. ADRs '
                      'are only one type of medication-related harm, as harm can also be caused by omitting to take '
                      'indicated medications.',
      'Peptic ulcer diseae':'Peptic ulcer disease(PUD) is a break in the innerlining of the stomach, the first '
                            'part of thesmall intestine, or sometimes the loweresophagus.An ulcer in the '
                            'stomach is called agastric ulcer, while one in the first part of the intestines is '
                            'aduodenal ulcer.The most common symptoms of a duodenal ulcer are waking at night '
                            'withupper abdominal painand upper abdominal pain that improves with eating.With a '
                            'gastric ulcer, the pain may worsen with eating.The pain is often described as '
                            'aburningor dull ache.Other symptoms includebelching, vomiting, weight loss, '
                            'orpoor appetite.About a third of older people have no symptoms.Complications may '
                            'includebleeding,perforation, andblockage of the stomach.Bleeding occurs in as '
                            'many as 15% of cases.',
      'AIDS':'Human immunodeficiency virus infection and acquired immunodeficiency syndrome(HIV/AIDS) is a spectrum '
             'of conditions caused byinfectionwith thehuman immunodeficiency virus(HIV),'
             'aretrovirus.Following initial infection a person may not notice any symptoms, or may experience a '
             'brief period ofinfluenza-like illness.Typically, this is followed by a prolonged period with no '
             'symptoms.If the infection progresses, it interferes more with theimmune system, increasing the risk '
             'of developing common infections such astuberculosis, as well as otheropportunistic infections, '
             'andtumorswhich are otherwise rare in people who have normal immune function.These late symptoms '
             'of infection are referred to as acquired immunodeficiency syndrome (AIDS).This stage is often also '
             'associated withunintended weight loss.',
      'Diabetes':'Diabetes mellitus(DM), commonly known asdiabetes, is a group ofmetabolic '
                 'disorderscharacterized by ahigh blood sugarlevel over a prolonged period of time.Symptoms '
                 'often includefrequent urination,increased thirstandincreased appetite.If left untreated, '
                 'diabetes can causemany health complications.Acutecomplications can includediabetic '
                 'ketoacidosis,hyperosmolar hyperglycemic state, or death.Serious long-term complications '
                 'includecardiovascular disease,stroke,chronic kidney disease,foot ulcers,damage to the '
                 'nerves,damage to the eyesandcognitive impairment.',
      'Gastroenteritis':'Gastroenteritisis a medical term forinflammationof thestomachandintestines. It '
                        'causesdiarrhea,vomitingandstomach pain. It usually happens because of infection by '
                        'avirusorbacteria.',
      'Bronchial Asthma':'Asthma(orAsthma bronchiale) is a disease that hurts theairwaysinside thelungs. It '
                         'causes thetissueinside the airways toswell. Asthma also causes the bands '
                         'ofmusclearound the airways to become narrow. This makes it hard for enough air to pass '
                         'through and for the person to breathe normally. Asthma also '
                         'causesmucus-makingcellsinside the airways to make more mucus than normal. This blocks '
                         'the airways, which are already very narrow during an asthma attack, and makes it even more '
                         'difficult to breathe.',
      'Hypertension':'Hypertension(HTNorHT), also known ashigh blood pressure(HBP), '
                     'is along-termmedical conditionin which theblood pressurein thearteriesis '
                     'persistently elevated.High blood pressure typically does not cause symptoms.Long-term high '
                     'blood pressure, however, is a major risk factor forstroke,coronary artery disease,'
                     'heart failure,atrial fibrillation,peripheral arterial disease,vision loss,'
                     'chronic kidney disease, anddementia.',
      'Migraine':'Migraine(UK:/ËˆmiËÉ¡reÉªn/,US:/ËˆmaÉª-/)is aprimary headache disordercharacterized '
                 'by recurrentheadachesthat are moderate to severe.Typically, episodes affect one side of the '
                 'head, are pulsating in nature, and last from a few hours to three days.Associated symptoms may '
                 'includenausea,vomiting, andsensitivity to light,sound, orsmell.The pain is generally '
                 'made worse by physical activity,although regular exercise may have prophylactic effects.Up to '
                 'one-third of people affected haveaura: typically a short period of visual disturbance that '
                 'signals that the headache will soon occur.Occasionally, aura can occur with little or no headache '
                 'following.',
      'Cervical spondylosis':'Spondylosisis the degeneration of thevertebral columnfrom any cause. In the more '
                             'narrow sense it refers to spinalosteoarthritis, the age-related wear and tear of the '
                             'spinal column, which is the most common cause of spondylosis. The degenerative process '
                             'in osteoarthritis chiefly affects the vertebral bodies, theneural foraminaand '
                             'thefacet joints(facet syndrome). If severe, it may cause pressure on thespinal '
                             'cordornerve rootswith subsequentsensoryormotordisturbances, '
                             'such aspain,paresthesia,imbalance, andmuscle weaknessin the limbs.',
      'Paralysis (brain hemorrhage)':'Intracerebral hemorrhage(ICH), also known ascerebral '
                                     'bleedandintraparenchymal bleed, is a sudden bleeding intothe tissues of '
                                     'the brain, into itsventricles, or into both.It is one kind of bleeding '
                                     'within theskulland is one kind ofstroke.',
      'Jaundice':'Jaundice, also known asicterus, is a yellowish or greenish pigmentation of theskinandwhites '
                 'of the eyesdue tohigh bilirubin levels.Jaundice in adults is typically a sign indicating the '
                 'presence of underlying diseases involving abnormalhememetabolism,liver dysfunction, '
                 'orbiliary-tractobstruction.The prevalence of jaundice in adults is rare, whilejaundice in '
                 'babiesis common, with an estimated 80% affected during their first week of life.The most '
                 'commonly associated symptoms of jaundice areitchiness,palefeces, anddark urine.',
      'Malaria':'Malariais amosquito-borne infectious diseasethat affects humans and other animals.Malaria '
                'causessymptomsthat typically includefever,tiredness,vomiting, andheadaches.In severe '
                'cases, it can causeyellow skin,seizures,coma, ordeath.Symptoms usually begin ten to '
                'fifteen days after being bitten by an infectedmosquito.If not properly treated, people may have '
                'recurrences of the disease months later.In those who have recently survived aninfection, '
                'reinfection usually causes milder symptoms.This partialresistancedisappears over months to '
                'years if the person has no continuing exposure to malaria.',
      'Chicken pox':'Chickenpox, also known asvaricella, is a highlycontagiousdisease caused by the '
                    'initialinfectionwithvaricella zoster virus(VZV).The disease results in a '
                    'characteristic skin rash that formssmall, itchy blisters, which eventually scab over.It '
                    'usually starts on the chest, back, and face.It then spreads to the rest of the body.Other '
                    'symptoms may includefever,tiredness, andheadaches.Symptoms usually last five to seven '
                    'days.Complications may occasionally includepneumonia,inflammation of the brain, '
                    'and bacterial skin infections.The disease is often more severe in adults than in '
                    'children.The incubation period is 10â€“21 days, 14â€“16 days, after which, a characteristic '
                    'rash appears.',
      'Dengue':'Dengue is a mosquito-borne, acute viral syndrome caused by any of the four serotypes of dengue virus '
               '(DENV). 1. In 2019, the World Health Organization designated dengue as one of the top 10 global '
               'health threats. 2. An estimated 50 million to 100 million symptomatic cases occur globally each year.',
      'Typhoid':'Typhoid fever, also known astyphoid, is a disease caused bySalmonellaserotype Typhi '
                'bacteria.Symptoms may vary from mild to severe, and usually begin 6 to 30 days after '
                'exposure.Often there is a gradual onset of a highfeverover several days.This is commonly '
                'accompanied by weakness,abdominal pain,constipation,headaches, and mild vomiting.Some people '
                'develop a skin rash withrose colored spots.In severe cases, people may experience '
                'confusion.Without treatment, symptoms may last weeks or months.Diarrheais uncommon.Other '
                'people may carry the bacterium without being affected, but they are still able to spread the disease '
                'to others.Typhoid fever is a type ofentericfever, along withparatyphoid fever.',
      'hepatitis A':'Hepatitis Ais an infectious disease of thelivercaused byHepatovirus A(HAV);it is a '
                    'type ofviral hepatitis.Many cases have few or no symptoms, especially in the young.The '
                    'time between infection and symptoms, in those who develop them, is between two and six '
                    'weeks.When symptoms occur, they typically last eight weeks and may include nausea, vomiting, '
                    'diarrhea,jaundice, fever, and abdominal pain.Around 10â€“15% of people experience a '
                    'recurrence of symptoms during the six months after the initial infection.Acute liver '
                    'failuremay rarely occur, with this being more common in the elderly.',
      'Hepatitis B':'Hepatitis Bis aninfectious diseasecaused by thehepatitis B virus(HBV) that affects '
                    'theliver;it is a type ofviral hepatitis.It can cause both acute andchronic '
                    'infection.Many people have no symptoms during the initial infection.In acute infection, '
                    'some may develop a rapid onset of sickness with vomiting,yellowish skin,tiredness, '
                    'dark urine, andabdominal pain.Often these symptoms last a few weeks and rarely does the '
                    'initial infection result in death.It may take 30 to 180 days for symptoms to begin.In those '
                    'who get infected around the time of birth 90% develop chronichepatitis Bwhile less than 10% '
                    'of those infected after the age of five do.Most of those with chronic disease have no '
                    'symptoms; however,cirrhosisandliver cancermay eventually develop.Cirrhosis or liver '
                    'cancer occur in about 25% of those with chronic disease.',
      'Hepatitis C':'Hepatitis Cis aninfectious diseasecaused by thehepatitis C virus(HCV) that primarily '
                    'affects theliver;it is a type ofviral hepatitis.During the initial infection people '
                    'often have mild or no symptoms.Occasionally a fever, dark urine, abdominal pain, andyellow '
                    'tinged skinoccurs.The virus persists in the liver in about 75% to 85% of those initially '
                    'infected.Early on chronic infection typically has no symptoms.Over many years however, '
                    'it often leads toliver diseaseand occasionallycirrhosis.In some cases, those with '
                    'cirrhosis will develop serious complications such asliver failure,liver cancer, ordilated '
                    'blood vessels in the esophagusandstomach.',
      'Hepatitis D':'Hepatitis Dis a type ofviralhepatitiscaused by thehepatitis delta virus(HDV), '
                    'a smallparticlethat are alike toviroidandvirusoid.',
      'Hepatitis E':'Hepatitis E has mainly afecal-oraltransmission that is similar tohepatitis A, '
                    'but the viruses are unrelated.',
      'Alcoholic hepatitis':'Alcoholic hepatitisishepatitis(inflammation of theliver) due to excessive intake '
                            'ofalcohol.Patients typically have a history of decades of heavy alcohol intake, '
                            'typically 8-10 drinks per day.It is usually found in association withfatty liver, '
                            'an early stage ofalcoholic liver disease, and may contribute to the progression of '
                            'fibrosis, leading tocirrhosis. Symptoms may present acutely after a large amount of '
                            'alcoholic intake in a short time period, or after years of excess alcohol intake. Signs '
                            'and symptoms of alcoholic hepatitis includejaundice(yellowing of the skin and eyes),'
                            'ascites(fluid accumulation in theabdominal cavity),fatigueandhepatic '
                            'encephalopathy(braindysfunction due toliver failure).Mild cases are '
                            'self-limiting, but severe cases have a high risk ofdeath. Severe cases may be treated '
                            'withglucocorticoids.',
      'Tuberculosis':'Tuberculosis(TB) is aninfectious diseaseusually caused byMycobacterium tuberculosis('
                     'MTB)bacteria.Tuberculosis generally affects thelungs, but can also affect other parts of '
                     'the body.Most infections show no symptoms, in which case it is known aslatent '
                     'tuberculosis.About 10% of latent infections progress to active disease which, '
                     'if left untreated, kills about half of those affected.The classic symptoms of active TB are a '
                     'chroniccoughwithblood-containingmucus,fever,night sweats, andweight loss.It was '
                     'historically calledconsumptiondue to the weight loss.Infectionof other organs can cause '
                     'a wide range of symptoms.',
      'Common Cold':'Thecommon cold, also known simply as acold, is aviralinfectious diseaseof theupper '
                    'respiratory tractthat primarily affects therespiratory mucosaof thenose,throat,'
                    'sinuses, andlarynx.Signs and symptoms may appear less than two days after exposure to the '
                    'virus.These may includecoughing,sore throat,runny nose,sneezing,headache, '
                    'andfever.People usually recover in seven to ten days,but some symptoms may last up to '
                    'three weeks.Occasionally, those with otherhealth problemsmay developpneumonia.',
      'Pneumonia':'Pneumoniais aninflammatorycondition of thelungprimarily affecting the small air sacs '
                  'known asalveoli.Symptoms typically include some combination ofproductiveor drycough,'
                  'chest pain,feveranddifficulty breathing.The severity of the condition is '
                  'variable.Pneumonia is usually caused byinfectionwithvirusesorbacteria, '
                  'and less commonly by othermicroorganisms. Identifying the responsible pathogen can be '
                  'difficult. Diagnosis is often based on symptoms andphysical examination.Chest X-rays, '
                  'blood tests, andcultureof the sputummay help confirm the diagnosis.The disease may be '
                  'classified by where it was acquired, such as community- or hospital-acquired or '
                  'healthcare-associated pneumonia.',
      'Dimorphic hemmorhoids(piles)':'Hemorrhoids(Piles) are blood vessels located in the smooth muscles of the '
                                     'walls of the rectum and anus. They are a normal part of the anatomy and are '
                                     'located at the junction where small arteries merge into veins. They are '
                                     'cushioned by smooth muscles and connective tissue and are classified by where '
                                     'they are located in relationship to the pectinate line, the dividing point '
                                     'between the upper 2/3 and lower 1/3 of the anus. This is an important anatomic '
                                     'distinction because of the type of cells that linehemorrhoid, and the nerves '
                                     'that provide sensation.',
      'Heart attack':'A heart attack occurs when one or more of your coronary arteries becomes blocked. Over time, '
                     'a buildup of fatty deposits, including cholesterol, form substances called plaques, '
                     'which can narrow the arteries (atherosclerosis). This condition, called coronary artery '
                     'disease, causes most heart attacks.',
      'Varicose veins':'Varicose veins are twisted, enlarged veins. Any superficial vein may become varicosed, '
                       'but the veins most commonly affected are those in your legs. That is because standing and '
                       'walking upright increases the pressure in the veins of your lower body.',
      'Hypothyroidism':'Hypothyroidism (underactive thyroid) is a condition in which your thyroid gland does not '
                       'produce enough of certain crucial hormones. Hypothyroidism may not cause noticeable symptoms '
                       'in the early stages. Over time, untreated hypothyroidism can cause a number of health '
                       'problems, such as obesity, joint pain, infertility and heart disease.',
      'Hyperthyroidism':'Hyperthyroidism (overactive thyroid) occurs when your thyroid gland produces too much of the '
                        'hormone thyroxine. Hyperthyroidism can accelerate your bodys metabolism, '
                        'causing unintentional weight loss and a rapid or irregular heartbeat. Several treatments are '
                        'available for hyperthyroidism. Doctors use anti-thyroid medications and radioactive iodine '
                        'to slow the production of thyroid hormones. Sometimes, hyperthyroidism treatment involves '
                        'surgery to remove all or part of your thyroid gland.',
      'Osteoarthristis':'Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. '
                        'It occurs when the protective cartilage that cushions the ends of your bones wears down over '
                        'time. Although osteoarthritis can damage any joint, the disorder most commonly affects '
                        'joints in your hands, knees, hips and spine.',
      'Arthritis':'Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of '
                  'arthritis are joint pain and stiffness, which typically worsen with age. The most common types of '
                  'arthritis are osteoarthritis and rheumatoid arthritis. Osteoarthritis causes cartilage â€” the '
                  'hard, slippery tissue that covers the ends of bones where they form a joint â€” to break down. '
                  'Rheumatoid arthritis is a disease in which the immune system attacks the joints, beginning with '
                  'the lining of joints.',
      '(vertigo) Paroymsal  Positional Vertigo':'Benign paroxysmal positional vertigo (BPPV) is one of the most '
                                                'common causes of vertigo â€” the sudden sensation that you are '
                                                'spinning or that the inside of your head is spinning. BPPVcauses '
                                                'brief episodes of mild to intense dizziness. It is usually triggered '
                                                'by specific changes in your heads position. This might occur when '
                                                'you tip your head up or down, when you lie down, or when you turn '
                                                'over or sit up in bed.',
      'Acne':'Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin '
             'cells. It causes whiteheads, blackheads or pimples. Acne is most common among teenagers, '
             'though it affects people of all ages. Effective acne treatments are available, but acne can be '
             'persistent. The pimples and bumps heal slowly, and when one begins to go away, others seem to crop up.',
      'Urinary tract infection':'A urinary tract infection (UTI) is an infection in any part of your urinary system '
                                'â€” your kidneys, ureters, bladder and urethra. Most infections involve the lower '
                                'urinary tract â€” the bladder and the urethra. Women are at greater risk of '
                                'developing aUTIthan are men. Infection limited to your bladder can be painful '
                                'and annoying. However, serious consequences can occur if aUTIspreads to your '
                                'kidneys.',
      'Psoriasis':'Psoriasis is a skin disease that causes red, itchy scaly patches, most commonly on the knees, '
                  'elbows, trunk and scalp. Psoriasis is a common, long-term (chronic) disease with no cure. It tends '
                  'to go through cycles, flaring for a few weeks or months, then subsiding for a while or going into '
                  'remission. Treatments are available to help you manage symptoms. And you can incorporate lifestyle '
                  'habits and coping strategies to help you live better with psoriasis.',
      'Impetigo':'Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects '
                 'infants and young children. It usually appears as reddish sores on the face, especially around the '
                 'nose and mouth and on the hands and feet. Over about a week, the sores burst and develop '
                 'honey-colored crusts.',
      'Hypoglycemia':'Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. '
                     'Glucose is your bodys main energy source. Hypoglycemia is often related to diabetes treatment. '
                     'But other drugs and a variety of conditions â€” many rare â€” can cause low blood sugar in '
                     'people who dont have diabetes'}
dict_={'Fungal infection':'bath twice ,use detol or neem in bathing water,keep infected area dry,use clean cloths ',
      'GERD':'avoid fatty spicy food ,avoid lying down after eating, maintain healthy weight,exercise ',
      'Chronic cholestasis':'cold baths,anti itch medicine,consult doctor, eat healthy ',
      'Drug Reaction':'stop irritation,consult nearest hospital,stop taking drug,follow up ',
      'Peptic ulcer diseae':'avoid fatty spicy food , consume probiotic food, eliminate milk ,limit alcohol ',
      'AIDS':'avoid open cuts , wear ppe if possible, consult doctor, follow up',
      'Diabetes':'have balanced diet, exercise ,consult doctor , follow up',
      'Gastroenteritis':'stop eating solid food for while,	try taking small sips of water,	rest	,ease back into eating',
      'Bronchial Asthma':'switch to loose cloothing, take deep breaths, get away from trigger, seek help',
      'Hypertension':'meditation, salt baths, reduce stress	,get proper sleep',
      'Migraine':'meditation,reduce stress,use poloroid glasses in sun,consult doctor',
      'Cervical spondylosis':'use heating pad or cold pack,	exercise,	take otc pain reliver, consult doctor',
      'Paralysis (brain hemorrhage)':'massage, eat healthy,	exercise,	consult doctor',
      'Jaundice':'drink plenty of water	, consume milk thistle	,eat fruits and high fiberous food,	medication',
      'Malaria':'Consult nearest hospital,avoid oily food,avoid non veg food,keep mosquitos out ',
      'Chicken pox':'use neem in bathing ,	consume neem leaves	,take vaccine	,avoid public places',
      'Dengue':'drink papaya leaf juice	,avoid fatty spicy food,	keep mosquitos ,away	keep hydrated',
      'Typhoid':'eat high calorie vegitables ,	antiboitic therapy,	consult doctor	,medication ',
      'hepatitis A':'Consult nearest hospital, wash hands through ,	avoid fatty spicy food	,medication',
      'Hepatitis B':'consult nearest hospital,	vaccination	,eat healthy.	medication',
      'Hepatitis C':'Consult nearest hospital,	vaccination	,eat healthy,	medication',
      'Hepatitis D':'consult doctor,	medication,	eat healthy,	follow up',
      'Hepatitis E':'consult doctor, medication,	eat healthy	,follow up',
      'Alcoholic hepatitis':'stop alcohol consumption,	rest,	consult doctor,	medication',
      'Tuberculosis':'cover mouth,	consult doctor	,medication	,rest',
      'Common Cold':'drink vitamin c rich drinks,	take vapour,	avoid cold food	,keep fever in check',
      'Pneumonia':'consult doctor,	medication,	rest,	follow up',
      'Dimorphic hemmorhoids(piles)':'avoid fatty spicy food,	consume witch hazel	,warm bath with epsom salt,	consume alovera juice',
      'Heart attack':'call ambulance,	chew or swallow asprin	,keep calm',
      'Varicose veins':'lie down flat and raise the leg high	use oinments	use vein compression	dont stand still for long ',
      'Hypothyroidism':'reduce stress,exercise,eat healthy,get proper sleep ',
      'Hyperthyroidism':'eat healthy	massage	use lemon balm	take radioactive iodine treatment',
      'Osteoarthristis':'acetaminophen	consult nearest hospital	follow up	salt baths',
      'Arthritis':'Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of '
                  'arthritis are joint pain and stiffness, which typically worsen with age. The most common types of '
                  'arthritis are osteoarthritis and rheumatoid arthritis. Osteoarthritis causes cartilage â€” the '
                  'hard, slippery tissue that covers the ends of bones where they form a joint â€” to break down. '
                  'Rheumatoid arthritis is a disease in which the immune system attacks the joints, beginning with '
                  'the lining of joints.',
      '(vertigo) Paroymsal  Positional Vertigo':'lie down flat and raise the leg high,	use oinments,	use vein compression,	dont stand still for long',
      'Acne':'bath twice	,avoid fatty spicy food	drink plenty of water,	avoid too many products',
      'Urinary tract infection':'drink plenty of water,	increase vitamin c intake,	drink cranberry juice	take probiotics',
      'Psoriasis':'wash hands with warm soapy water ,stop bleeding using pressure,consult doctor,salt baths ',
      'Impetigo':'Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects '
                 'infants and young children. It usually appears as reddish sores on the face, especially around the '
                 'nose and mouth and on the hands and feet. Over about a week, the sores burst and develop '
                 'honey-colored crusts.',
      'Hypoglycemia':'lie down on side,	check in pulse,	drink sugary drinks,	consult doctor'}
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print(tag, "msg-", msg)
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."



app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    if msg=="No" or msg=="Predict" or msg=="predict" or msg=="no":
        array = df.values
        array = np.asarray(array).astype(np.float32)
        predictions = disease_model.predict(array)
        predicted_class = np.argmax(predictions)
        d = disease_names[predicted_class]
        global disease
        disease=d
        text="You have {} , type 'Describe' to get full info , type 'Precations' if you need any".format(d)
        return text
    
    if msg=="Describe" or msg=="describe":
        if dict.get(disease) is not None:
            return dict.get(disease)
        else:
            return "Discription Not found :-("
    if msg=="Precations" or msg=="precations":
        if dict_.get(disease) is not None:
            return dict_.get(disease)
        else:
            return "Precations Not found :-("
    if msg in column_names:
        df.at[0,msg] = 1
    input = msg
    return get_response(input)


if __name__ == '__main__':
    app.run()


#0 0 0 0 0 