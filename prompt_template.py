SAME_EVENT_JUDGMENT_PROMPT = """
It is known that event triggers are important informations that can indicate an event occurrence. Now given two event statements along with their corresponding event trigger words (provided as keywords, maybe none), determine whether the statements describe the same event based on the trigger words and event participants (e.g., entities, roles, context, numbers), If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
Health Secretary Andrew Lansley said the pledges were part of the government 's `` bold new approach to public health , '' avoiding new legislation and relying on self-policing by industry .

### Keywords
said

### Input 2
The pledge was unlikely to dampen the intensity of protests .

### Keywords


### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
The initiative aims to lower salt content in food and restrict the promotion of alcoholic drinks .

### Keywords
aims

### Input 2
The scaled-back version is aimed at winning the support of China and Russia , which oppose sanctions .

### Keywords


### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Health Secretary Andrew Lansley said the pledges were part of the government 's `` bold new approach to public health , '' avoiding new legislation and relying on self-policing by industry .

### Keywords
said

### Input 2
`` Most of them are still giving it out , '' Merewood told Reuters Health .

### Keywords


### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
no-fly zone was imposed around the reactors .

### Keywords
imposed

### Input 2
Last week , the head of U.S. Joint Forces Command said the Pentagon could implement a no-fly zone ` within a couple of days . '

### Keywords
said

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Despite pleas for calm , residents rushed to shops in Tokyo to stock up on supplies .

### Keywords
Despite pleas,rushed,stock up on

### Input 2
Most shops were closed .

### Keywords
closed

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Last flu season , only 19 percent of H1N1 viruses tested were Tamiflu-resistant , Dr. Nila Dharan and colleagues at the CDC reported .

### Keywords
tested

### Input 2
CDC researchers said 98 percent of all flu samples from the H1N1 strain were resistant to Roche AG 's Tamiflu , a pill that can both treat flu and prevent infection .

### Keywords
said,were

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
The government retook the district last week after fierce shelling .

### Keywords
retook

### Input 2
Heavy shelling and rocket fire were reported overnight and into Tuesday as the government attempted to take back the seized districts .

### Keywords
were,reported,attempted to

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
{input_1}

### Keywords
{keywords_1}

### Input 2
{input_2}

### Keywords
{keywords_2}

### The determination of whether the above two statements describing the same event is
"""


SAME_EVENT_JUDGMENT_PROMPT_TMP = """
It is known that event triggers are important informations that can indicate an event occurrence. Now given two event statements along with their corresponding event trigger words (provided as keywords, maybe none), determine whether the statements describe the same event based on the trigger words and event participants (e.g., entities, roles, context, numbers), If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
The influenza strain that has struck Mexico and the United States involves , in many cases , a never-before-seen strain of the H1N1 virus .

### Keywords
struck

### Input 2
The influenza strain is an H1N1 , the same family as one of the seasonal flu viruses now circulating .

### Keywords
is

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
As the death toll from swine flu in Mexico rises to more than 100 people , governments around the world are on high alert for a possible flu pandemic .

### Keywords
death,high alert

### Input 2
If the confirmed deaths are the first signs of a pandemic , then cases are probably incubating around the world by now , said Dr Michael Osterholm , a flu expert at the University of Minnesota .

### Keywords
confirmed,death

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Roche , the maker of Tamiflu , said it was prepared to immediately deploy a stockpile of the drug if requested .

### Keywords
said

### Input 2
Has stocks of 2.5 million doses of Tamiflu - enough for a quarter of the population .

### Keywords


### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
If the confirmed deaths are the first signs of a pandemic , then cases are probably incubating around the world by now , said Dr Michael Osterholm , a flu expert at the University of Minnesota .

### Keywords
confirmed,death

### Input 2
Given how quickly flu can spread , there might be cases incubating around the world already , said Dr Michael Osterholm at the University of Minnesota : `` Hundreds and thousands of travellers come in and out -LRB- of Mexico -RRB-

### Keywords


### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Up to 169 people are believed to have died in the outbreak - all but one of them in Mexico .

### Keywords
are,rushed,died in the outbreak

### Input 2
Twenty people are known to have died in Mexico so far out of a total of 1,004 reported cases , and 48 more deaths are thought to be attributable to the outbreak .

### Keywords
known,died in

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
They have not been hospitalised , and the state described their illnesses as mild .

### Keywords
hospitalised,illnesses as

### Input 2
None of them are seriously ill .

### Keywords


### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Swine flu : pandemic threat alert raised The British couple being treated for swine flu have been named , as fear of a pandemic increase and the death toll in Mexico continues to rise .

### Keywords


### Input 2
State health officials said yesterday they had confirmed swine flu in a married couple living in the central part of the state after the husband visited Mexico .

### Keywords
said

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
{input_1}

### Keywords
{keywords_1}

### Input 2
{input_2}

### Keywords
{keywords_2}

### The determination of whether the above two statements describing the same event is
"""


RELATION_STATEMENTS_SUMMARY_PROMPT = """
All the given statements describe a single event. Based on the complete set of these statements, generate a concise summary of this event.

#################

### Input Set 1
At least 11 Brazilian peacekeepers were killed , Brazil 's military has said , according to the AFP news agency .
1200 Several UN international peacekeepers are reported to be among the dead , including people from Brazil and Jordan .
He says a total of 15 UN staff are now confirmed as dead - 11 Brazilian peacekeepers , as well as three Jordanians , one Argentine and one Chadian who were police officers .
2112 Susana Malcorra , the head of the UN department of field support , confirms that at least 14 UN personnel have been killed - three Jordanian and 11 Brazilian peacekeepers , and one Haitian civilian .

### The Summarization of the above Input Set 1 is
1200 Several UN international peacekeepers are reported to be dead , and a total of 15 UN staff are now confirmed as dead - 11 Brazilian peacekeepers , as well as three Jordanians , one Argentine and one Chadian who were police officers .

#################

### Input Set 2
Mr Haas said there were also `` serious problems '' with the enforcement of building codes in Haiti .
`` People are skimping on cement to try to cut costs , putting a lot of water in , building too thin , and you end up with a structure that 's innately weaker , '' said Mr Haas , who was on his way to Haiti to help assess the safety of damaged buildings .
Concrete blocks are being made in people 's backyards and dried out in the sun There are also significant problems with the quality of building materials used , says Peter Haas , head of the Appropriate Infrastructure Development Group , a US-based non-profit group that has been working in Haiti since 2006 .
At the time , Haitian authorities blamed poor construction for the accidents .
Even before the quake , Haiti 's building safety record was poor .
Experts say it is no surprise that shoddy construction contributed to the level of destruction in Haiti following Tuesday 's earthquake . Peter Haas said there are also significant problems with the quality of building materials used  Mr Haas said there were also `` serious problems '' with the enforcement of building codes in Haiti .
`` Concrete blocks are being made in people 's backyards and dried out in the sun , '' he said .

### The Summarization of the above Input Set 2 is
Experts say it is no surprise that shoddy construction contributed to the level of destruction in Haiti following Tuesday 's earthquake . Peter Haas said there are significant problems with the quality of building materials used and Mr Haas said there were also `` serious problems '' with the enforcement of building codes in Haiti .

#################

### Input Set 3
{Input_Set}

### The Summarization of the above Input Set 3 is
"""


RELATION_STATEMENTS_SUMMARY_PROMPT_TMP = """
All the given statements describe a single event. Based on the complete set of these statements, generate a concise summary of this event.

#################

### Input Set 1
Dr. David Weinstock of the Dana-Farber Cancer Institute and Dr. Gianna Zuccotti of Brigham and Women 's Hospital , both in Boston , said the quick spread of Tamiflu-resistant flu had surprised doctors .
Last flu season , only 19 percent of H1N1 viruses tested were Tamiflu-resistant , Dr. Nila Dharan and colleagues at the CDC reported .
There is no indication the two other types of season flu now circulating , H3N2 and influenza B , resist the effects of Tamiflu and the CDC recommends using a cocktail of flu drugs in patients now .
CDC researchers said 98 percent of all flu samples from the H1N1 strain were resistant to Roche AG 's Tamiflu , a pill that can both treat flu and prevent infection .
Flu already resists two older drugs , rimantadine and amantadine .

### The Summarization of the above Input Set 1 is
Flu already resists two older drugs , rimantadine and amantadine . CDC researchers said 98 percent of all flu samples from the H1N1 strain were resistant to Roche AG 's Tamiflu. The quick spread of Tamiflu-resistant flu had surprised Dr. David and  Dr. Gianna Zuccotti . There is no indication the two other types of season flu now circulating , H3N2 and influenza B , resist the effects of Tamiflu and the CDC recommends using a cocktail of flu drugs in patients now.

#################

### Input Set 2
The government has declared a `` state of health alert '' .
The health ministry has declared a health alert .
He said the government had alerted all borders to be on alert for potential carriers of the disease .
The government has declared a health emergency that would release funds , which could be used to help deal with the situation .
Health authorities have been ordered to watch for an increase in respiratory illnesses and to promote vaccinations and preventative hygiene habits among health workers and the public .
The administration says they have health personnel at ports looking out for people who may have the disease .
The authorities have also warned against non-vital travel to Mexico and the US , and border officials are on alert to monitor passengers arriving by land who have flu-like symptoms .
Officials are monitoring visitors who have come from infected areas such as the US , Canada , Israel , Spain and the UK .

### The Summarization of the above Input Set 2 is
The government has declared a `` state of health alert '' .The health ministry has been ordered to watch for an increase in respiratory illnesses and to promote vaccinations and preventative hygiene habits . Officials are monitoring visitors who have come from infected areas such as the US , Canada , Israel , Spain and the UK . The authorities have also warned against non-vital travel to Mexico and the US , and border officials are on alert to monitor passengers arriving by land who have flu-like symptoms . The administration has declared that they would release funds to look out for people who may have the disease .

#################

### Input Set 3
{Input_Set}

### The Summarization of the above Input Set 3 is
"""


RELATION_CLUSTER_SPLIT_PROMPT = """
Given two event statements, determine whether the statements describe the same event. If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
1943 US military officials say aid flights to Haiti have resumed after being suspended because of overcrowding at Port-au-Prince airport - Associated Press .

### Input 2
Wyclef Jean says he 's now planning another TV fundraiser featuring Black Eyed Peas on 5 February .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
The US military stopped the flights to Florida on Wednesday .

### Input 2
Meanwhile , the US Federal Aviation Authority said it had stopped civilian flights to Haiti at the Haitian government 's request because there was not enough space on the ground for more planes and only limited fuel for them to leave .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Contributors Minogue and Williams previously duetted on the single Kids Everybody Hurts , the all-star single recorded to raise money for victims of the Haiti earthquake , will be released on 7 February , it has been confirmed .

### Input 2
JLS singer Ortise Williams , who lost relatives in the 12 January earthquake , said : `` The tragedy is very close to my heart .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
The US military has begun airdropping food and water supplies into earthquake-hit Haiti, despite earlier concerns about the risk and challenges with aid distribution due to airport congestion.

### Input 2
The US military has begun distributing aid in Haiti , the Associated Press reports .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
One thousand metric tons of ready-to-eat meals will arrive in Port-au-Prince on 27 January .

### Input 2
Thousands of people joined open-air church services in Port-au-Prince , Leogane - the epicentre of the earthquake - and elsewhere on Sunday .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements describing the same event is
"""


RELATION_CLUSTER_SPLIT_PROMPT_TMP = """
Given two event statements, determine whether the statements describe the same event. If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
Several drugs were found in Michael Jackson 's body.

### Input 2
The Black Eyed Peas have withdrawn from the Michael Jackson tribute concert in Cardiff, with CEO Chris Hunt stating that they are removing the group from the event but looking forward to featuring other artists.

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Michael Jackson received a 10 milligram dose of Diazepam at 1:30 am and a dose of Propofol diluted with Lidocaine at 10:40 am on the day of his death due to sleep difficulties.

### Input 2
Hundreds of Jackson fans lined the street .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Jackson was rehearsing for his 50-date London residency when he died 25 June 2009.

### Input 2
The singer died suddenly in June of 2009 from a prescription drug overdose at age 50 , weeks before beginning a set of concerts .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
The prosecution team claimed Conrad Murray was an incompetent physician who used an anesthetic called Propofol without the proper safeguards .

### Input 2
The doctor is alleged to have administered a lethal dose of Propofol and other drugs , which resulted in the popstar 's death on 25 June .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Hundreds of Michael Jackson fans gathered outside the court in downtown Los Angeles, where they waited anxiously for the verdict. Many had tickets to watch inside, but most were unable to stay, leading to tension and eventual police intervention as fans blocked the pavements.

### Input 2
Entertainers, world leaders, and fans have continued to pay tribute to the star, praising him as the consummate entertainer whose contributions and legacy will be felt worldwide.

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements describing the same event is
"""


SAME_EVENT_CLUSTER_SPLIT_PROMPT_TMP = """
Given two event statements, determine whether the two statements are the same event. If they describe the same event, and the semantics are basically identical, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
The US effort in Haiti has drawn criticism from some Latin American leaders, who claim the US lacks close rapport with the territory and international organizations. A senior Italian official also criticized the Haiti earthquake relief operation, saying it could have been managed better. US troops have arrived in Haiti, and a US diplomat dismissed the criticisms, stating Washington's goal is to provide aid without being distracted by political remarks.

### Input 2
Guido Bertolaso , who is in Haiti to co-ordinate relief efforts , also criticised what he saw as the presence of too many American soldiers .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
A seven-year-old boy named Charlie raised over £190,000 for victims of the Haiti earthquake by doing a sponsored bike ride around his local park. He received donations from as far away as Hong Kong and New Zealand after people read his appeal on the JustGiving website, far exceeding his goal of £500 in sponsorship.

### Input 2
A seven-year-old boy is raising at least $35,000 for victims of the Haiti earthquake through a sponsored bike ride to support Unicef's efforts in providing food, water, and healthcare for children in Haiti.

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
The quake , which struck about 15km -LRB- 10 miles -RRB- south-west of Port-au-Prince , was quickly followed by two strong aftershocks of 5.9 and 5.5 magnitude .

### Input 2
The center of the quake hit near Port-au-Prince and was quickly followed by two strong aftershocks of 5.9 and 5.5 on the Richter scale .

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements are the same event is
"""


SAME_EVENT_CLUSTER_SPLIT_PROMPT = """
Given two event statements, determine whether the statements describe the same event. If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
EU health commissioner , Androulla Vassiliou , said there are reports of suspected cases in Denmark , Sweden , Greece , the Czech Republic , Germany , Italy and Ireland .

### Input 2
French health ministry officials said four possible cases of swine flu were under investigation : a family of three in the Nord region and a woman in the Paris region .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
A spokesman for NHS Direct said the advice line had received almost 1,400 calls about suspected swine flu cases .

### Input 2
NHS Direct has received more than 200 potential cases of swine flu in the past 24 hours , James Sturcke reports .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
The H1N1 swine flu virus is disproportionately affecting younger people, unlike seasonal flu which typically targets the elderly. While the elderly have been less likely to catch the infection, younger individuals, including children and young adults, have been most affected. The virus shows a different pattern of illness and impact compared to seasonal influenza.

### Input 2
Younger people were probably hit harder by the 1918 flu virus because their immune systems over-reacted .

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements are the same event is
"""

SAME_CLUSTER_SUMMARIZE_PROMPT = """Instruction
All the statements below describe the same single event. Each line is a specific detail about that event. Please analyze all the statements and provide a brief, coherent summary of the overall main event.

#################
Example:

### Statements 1:
Around 400,000 people will be relocated to tent villages outside the capital, with camps being constructed soon to address the situation.
Efforts to set up temporary tent camps began the day after the earthquake, with hundreds of people camping in the open. Mr. Gascon mentions a large number of tents are needed, with 40,000 more on the way from Panama by ship. Charles Clermont, a Haitian official, is responsible for building the mass tent cities to house thousands of refugees. This is the first proper tent encampment since the earthquake.
Efforts to set up temporary tent camps began the day after the earthquake, with hundreds of people camping in the open. Mr. Gascon mentions a large number of tents are needed, with 40,000 more on the way from Panama by ship. Charles Clermont, a Haitian official, is responsible for building the mass tent cities to house thousands of refugees. This is the first proper tent encampment since the earthquake.
About 400,000 survivors will be moved to tented villages outside the capital , Port-au-Prince , with 100,000 people initially being sent to 10 settlements near the suburb of Croix Des Bouquets , Interior Minister Paul Antoine Bien-Aime announced .
Haiti is planning to house 400,000 earthquake survivors in new tented villages outside the capital , Port-au-Prince , officials have announced .
Authorities in Haiti have announced plans to house 400,000 earthquake survivors in tented villages outside the capital , Port-au-Prince .
At least 500,000 people are currently living outdoors in 447 improvised camps in Port-au-Prince , according to the International Organisation for Migration -LRB- IOM -RRB- .
Hundreds of thousands of people have already left the city for tented settlements or family homes in other parts of Haiti .
Millions of people remain in need after Haiti 's earthquake , and plans are being made to house 400,000 survivors in new tented villages outside the capital .
Efforts to set up temporary tent camps began the day after the earthquake, with hundreds of people camping in the open. Mr. Gascon mentions a large number of tents are needed, with 40,000 more on the way from Panama by ship. Charles Clermont, a Haitian official, is responsible for building the mass tent cities to house thousands of refugees. This is the first proper tent encampment since the earthquake.
The Haitian government wants to relocate some 400,000 people , currently in makeshift camps across the capital , to temporary tent villages outside the city .
On Thursday , Mr Bien-Aime said public buses had already been sent out to take survivors in Port-au-Prince to the south and north of the country , where tented settlements able to accommodate 10,000 people each would eventually be built .
As the relief operation continues , aid workers have criticised Haitian government plans to relocate hundreds of thousands of people from the capital , Port-au-Prince , to large camps outside the city .
Haiti starts to move quake victims Quake survivors moved out of cities Haiti has started moving its earthquake survivors to camps outside the capital , where they should be safer .
People are being moved outside the capital to a place called Croix Des Bouquets, where camps are being built. The camps are not yet constructed but will be built soon.

### Based on the statements 1 above, write a concise summary of the main event part:
About 400,000 survivors will be moved to tented villages outside the capital , Port-au-Prince , with 100,000 people initially being sent to 10 settlements near the suburb of Croix Des Bouquets , Interior Minister Paul Antoine Bien-Aime announced .

#################
Your task:

### Statements 2:
{statements}

### Based on the statements 2 above, write a concise summary of the main event part:
"""

SAME_CLUSTER_SUMMARIZE_PROMPT_TMP = """Instruction
All the statements below describe the same single event. Each line is a specific detail about that event. Please analyze all the statements and provide a brief, coherent summary of the overall main event.

#################
Example:

### Statements 1:
Eleven confirmed infections in the US
In the US , 11 people are now known to have been infected with the new strain - seven people in California , two in Texas , and two in Kansas .
At least two more cases have been confirmed in Kansas , bringing the US total to 11 .
Besser said the cases are now in 11 states , up from 10 yesterday .

### Based on the statements 1 above, write a concise summary of the main event part:
11 people in US are now known to have been infected with the new strain : seven people in California , two in Texas , and two in Kansas .

#################
Your task:

### Statements 2:
{statements}

### Based on the statements 2 above, write a concise summary of the main event part:
"""


DAY_SUMMARIZE_PROMPT = """Instruction
All of the following statements describe events that occurred on a certain day. Each line is a specific detail about one event. All the statements are redundant. Please streamline them and output each simplified event statement on a separate line.

#################

### Statements 1:
The World Health Organization (WHO) has declared the H1N1 swine flu virus a global pandemic, marking the first influenza pandemic in 41 years. The outbreak, which began in Mexico, has spread to at least 41 countries and is now classified as a public health emergency of international concern. The WHO raised the pandemic alert level to phase 6, indicating widespread transmission across multiple regions, including the Americas and Australia. This declaration signifies that the virus is spreading easily between people and could not be contained, prompting increased global efforts to develop treatments and manage the crisis.
The World Health Organization (WHO) has raised its pandemic alert level to 5 on a 6-level scale, indicating that a pandemic is imminent. This decision follows rapid developments including the spread of H1N1 flu, new cases in Peru and Switzerland, and Mexico's shutdown. While the WHO has not yet moved to the highest level of 6, it emphasizes the global threat posed by the virus, particularly in vulnerable communities.
The World Health Organization declared a global influenza pandemic, marking the first of the 21st century, and advised governments to prepare for a long-term battle against the H1N1 virus, which has spread to 74 countries and is circulating globally.
Governments worldwide have implemented strict health measures to prevent the spread of swine flu, including enhanced airport screenings, thermal scanners, and monitoring of travelers from affected countries like Mexico, the US, and Canada. These efforts involve setting up field hospitals, isolation units, and quarantine centers, suspending flights to and from Mexico, advising against non-essential travel, and conducting thorough checks on passengers and their luggage. Emergency task forces and joint prevention systems have been established to manage the outbreak and ensure public health safety.
The new swine flu strain primarily causes mild symptoms, with nausea and stomach pain being the most common, while only a small minority experience serious illness, particularly those with pre-existing health conditions. Symptoms are similar to those of seasonal flu, and the World Health Organization provides the same guidance for caring for suspected cases as it does for seasonal flu.
Several countries have imposed import bans on pork and pork products from Mexico, the United States, and other nations affected by swine flu, citing concerns over the virus's spread. China and others have suspended pork imports, increased health inspections at borders, and culled pigs in some regions. The global pork industry is facing trade restrictions despite warnings that cooked pork is safe to eat.

### Based on Statements 1, all simplified event statements placed on separate lines are:
The World Health Organization (WHO) has declared the H1N1 swine flu virus a global pandemic, and has spread to at least 41 countries.
The WHO raised the pandemic alert level from 5 to phase 6.
The H1N1 virus has spread to 74 countries and is circulating globally.
Governments worldwide have implemented strict health measures to prevent the spread of swine flu, including enhanced airport screenings, thermal scanners, and monitoring of travelers from affected countries like Mexico, the US, and Canada. These efforts involve setting up field hospitals, isolation units, and quarantine centers, suspending flights to and from Mexico, advising against non-essential travel, and conducting thorough checks on passengers and their luggage. Emergency task forces and joint prevention systems have been established to manage the outbreak and ensure public health safety.
The World Health Organization provides the same guidance for caring for suspected cases as it does for seasonal flu.
Several countries have imposed import bans on pork and pork products from Mexico, the United States, and other nations affected by swine flu. China and others have suspended pork imports, increased health inspections at borders, and culled pigs in some regions.

#################

### Statements 2:
Mexico's government has implemented strict public health measures to prevent the spread of a disease, including banning traditional greetings like handshakes and kisses, closing public venues such as football matches, museums, and theaters, and advising people to avoid crowded places and share food or personal items. Residents in Mexico City have been ordered to maintain a distance of at least six feet from others and avoid physical contact, marking a significant change from customary social practices.
Mexico has confirmed 20 deaths from the H1N1 swine flu virus, with 40 additional possible fatalities and 1,004 reported cases. The government is investigating dozens more deaths, with all but one of the deaths believed to have occurred in Mexico.
The international community has decided not to raise the global pandemic alert level, with officials from the UN and WHO stating there are no imminent plans to declare an international public health emergency.
Authorities stated that Mexico has sufficient antiviral medicine to treat approximately 1,000 suspected cases of the outbreak, as confirmed by Health Minister Jose Angel Cordova.

### Based on Statements 2, all simplified event statements placed on separate lines are:
Mexico's government has implemented strict public health measures to prevent the spread of a disease, including banning traditional greetings like handshakes and kisses, closing public venues such as football matches, museums, and theaters, and advising people to avoid crowded places and share food or personal items.
Residents in Mexico City have been ordered to maintain a distance of at least six feet from others and avoid physical contact.
Mexico has confirmed 20 deaths from the H1N1 swine flu virus, with 40 additional possible fatalities and 1,004 reported cases.
The international community has decided not to raise the global pandemic alert level.
Health Minister Jose Angel Cordova confirmed that Mexico has sufficient antiviral medicine to treat approximately 1,000 suspected cases of the outbreak.

#################

### Statements 3:
{statements}

### Based on Statements 3, all simplified event statements placed on separate lines are:
"""


DAY_SUMMARIZE_PROMPT_TMP = """Instruction
All of the following statements describe events that occurred on a certain day. Each line is a specific detail about one event. All the statements are redundant. Please streamline them and output each simplified event statement on a separate line.

#################

### Statements 1:
A powerful 7.0-magnitude earthquake struck Haiti, causing unprecedented devastation. The quake, Haiti's worst in two centuries, killed an estimated 200,000 people, left 1.5 million homeless, and destroyed much of the country's infrastructure, including Port-au-Prince. Widespread destruction, collapsed buildings, and trapped victims were reported, with the disaster described as one of the worst in recent history.
A significant portion of Jacmel, Haiti, has been destroyed in the earthquake, with estimates suggesting at least 20% of the city's buildings are collapsed. The city, home to 50,000 people, is still assessing the full extent of the damage, and Unicef has confirmed the destruction and called for further assessment and response.
A devastating earthquake struck Port-au-Prince, leaving at least one million people homeless and causing widespread destruction, including collapsed buildings, significant car damage, and many people crying for help, bleeding, and without assistance.
The United Nations peacekeeping mission in Haiti is dealing with a significant loss of personnel following a powerful earthquake. The UN headquarters and facilities in Port-au-Prince were severely damaged, resulting in the disappearance of a large number of staff. Estimates suggest between 100 and 150 UN personnel are still missing, with up to 200 unaccounted for, including the civilian head of the mission, Hedi Annabi, who is feared dead. The earthquake has caused serious damage to UN installations, and many are believed to be buried under the rubble.
The earthquake in Haiti is the worst in two centuries, causing reports of a substantial number of deaths.
A major earthquake has caused widespread devastation, with fears that thousands of people may have died. Estimates of the death toll range from 50,000 to at least 200,000, though official numbers are still unclear. The disaster has also left many homeless and has resulted in significant damage to infrastructure, with rescue workers warning of a potentially high death toll.
International aid teams, including those from the United States, Taiwan, and the Caribbean Community (Caricom), are actively traveling to Haiti to provide rescue and humanitarian relief following a disaster. Emerson Tan, a volunteer aid worker, is part of a team working to reach Haiti, with US teams already en route with specialized rescue equipment and efforts underway.
An earthquake has caused significant concern and shock in Brazil, home to the Brazilian army's large UN contingent in Haiti. The UN peacekeeping mission in Haiti, Minustah, reports that about 100 or more of its staff are still unaccounted for after buildings collapsed.
The United States is providing full support to Haiti in its efforts to rescue people trapped in the rubble and deliver humanitarian aid, including food, water, and medicine. General Douglas Fraser, head of US Southern Command, stated they are doing everything possible to speed up the aid delivery. The Caribbean Community has also pledged assistance to Haiti.

### Based on Statements 1, all simplified event statements placed on separate lines are:
A powerful 7.0-magnitude earthquake, the worst in two centuries, struck Haiti. It killed an estimated 200,000 people, left 1.5 million homeless, and destroyed much of the country's infrastructure, including Port-au-Prince.
Unicef has confirmed the destruction and called for further assessment and response.
The UN headquarters and facilities in Port-au-Prince were severely damaged, resulting in the disappearance of a large number of staff. Estimates suggest between 100 and 150 UN personnel are still missing, with up to 200 unaccounted for, including the civilian head of the mission, Hedi Annabi, who is feared dead.
Estimates of the death toll range from 50,000 to at least 200,000, though official numbers are still unclear.
International aid teams, including those from the United States, Taiwan, and the Caribbean Community (Caricom), are actively traveling to Haiti to provide rescue and humanitarian relief following a disaster.
The Brazilian army's large UN contingent is destroyed in Haiti. The UN peacekeeping mission in Haiti, Minustah, reports that about 100 or more of its staff are still unaccounted for after buildings collapsed.
The United States is providing full support to Haiti in its efforts to rescue people trapped in the rubble and deliver humanitarian aid, including food, water, and medicine. The Caribbean Community has also pledged assistance to Haiti.

#################

### Statements 2:
Gate and the Hanging Gardens, has suffered extensive contamination and irreversible damage, including chemical spills, military vehicle traffic, and the importation of foreign materials that will permanently contaminate the site. Dr. Curtis emphasizes that Iraq lacks the resources to repair the damage and that an international effort is necessary to address the widespread destruction.
The archaeological site of Babylon has been severely damaged and contaminated during coalition forces' occupation, with major structures like the Ishtar Gate and ziggurat suffering significant destruction. Heavy vehicles, chemicals, and imported materials have caused irreversible harm, including broken bricks inscribed with Nebuchadnezzar's name and soil contamination. Dr. John Curtis of the British Museum calls for an international investigation, criticizing the coalition's actions as reckless and avoidable, which have jeopardized the preservation of this crucial archaeological treasure.
The invasion of Iraq is believed to have bolstered al-Qaida's propaganda, recruitment, and fundraising efforts while providing a training ground for Islamist militants. A report by the National Intelligence Council warns that terrorists trained in Iraq may become a successor generation to al-Qaida, replacing those who trained in Afghanistan, and could pose a global threat, including the use of biological or chemical weapons. A CIA thinktank also notes that the chaos in Iraq is fostering a new generation of terrorists likely to replace al-Qaida as a major global threat.

### Based on Statements 2, all simplified event statements placed on separate lines are:
The archaeological site of Babylon has been severely damaged and contaminated during coalition forces' occupation, with major structures like the Ishtar Gate and ziggurat suffering significant destruction. Heavy vehicles, chemicals, and imported materials have caused irreversible harm, including broken bricks inscribed with Nebuchadnezzar's name and soil contamination.
Dr. John Curtis of the British Museum calls for an international investigation, criticizing the coalition's actions as reckless and avoidable, which have jeopardized the preservation of this crucial archaeological treasure.
The invasion of Iraq is believed to have bolstered al-Qaida's propaganda, recruitment, and fundraising efforts while providing a training ground for Islamist militants.
A report by the National Intelligence Council warns that terrorists trained in Iraq may become a successor generation to al-Qaida, replacing those who trained in Afghanistan, and could pose a global threat, including the use of biological or chemical weapons.
A CIA thinktank also notes that the chaos in Iraq is fostering a new generation of terrorists likely to replace al-Qaida as a major global threat.

#################

### Statements 3:
{statements}

### Based on Statements 3, all simplified event statements placed on separate lines are:
"""
