# Predict Students' Dropout and Academic Success - Description du Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## Aperçu

Ce dataset contient des données sur les étudiants inscrits dans divers programmes de premier cycle d'un établissement d'enseignement supérieur au Portugal. Il permet de prédire l'abandon scolaire et la réussite académique.

- **Nombre d'observations:** 4424
- **Nombre de variables:** 37
- **Variable cible:** Target (Dropout, Enrolled, Graduate)

---

## Variables Catégorielles

### Marital Status (État civil)
| Code | Valeur |
|------|--------|
| 1 | Single (Célibataire) |
| 2 | Married (Marié) |
| 3 | Widower (Veuf) |
| 4 | Divorced (Divorcé) |
| 5 | Facto union (Union de fait) |
| 6 | Legally separated (Séparé légalement) |

### Application Mode (Mode de candidature)
| Code | Valeur |
|------|--------|
| 1 | 1st phase - general contingent |
| 2 | Ordinance No. 612/93 |
| 5 | 1st phase - special contingent (Azores Island) |
| 7 | Holders of other higher courses |
| 10 | Ordinance No. 854-B/99 |
| 15 | International student (bachelor) |
| 16 | 1st phase - special contingent (Madeira Island) |
| 17 | 2nd phase - general contingent |
| 18 | 3rd phase - general contingent |
| 26 | Ordinance No. 533-A/99, item b2 (Different Plan) |
| 27 | Ordinance No. 533-A/99, item b3 (Other Institution) |
| 39 | Over 23 years old |
| 42 | Transfer |
| 43 | Change of course |
| 44 | Technological specialization diploma holders |
| 51 | Change of institution/course |
| 53 | Short cycle diploma holders |
| 57 | Change of institution/course (International) |

### Application Order (Ordre de candidature)
Valeur entre 0 et 9 (0 = premier choix, 9 = dernier choix)

### Course (Programme d'études)
| Code | Valeur |
|------|--------|
| 33 | Biofuel Production Technologies |
| 171 | Animation and Multimedia Design |
| 8014 | Social Service (evening attendance) |
| 9003 | Agronomy |
| 9070 | Communication Design |
| 9085 | Veterinary Nursing |
| 9119 | Informatics Engineering |
| 9130 | Equinculture |
| 9147 | Management |
| 9238 | Social Service |
| 9254 | Tourism |
| 9500 | Nursing |
| 9556 | Oral Hygiene |
| 9670 | Advertising and Marketing Management |
| 9773 | Journalism and Communication |
| 9853 | Basic Education |
| 9991 | Management (evening attendance) |

### Daytime/Evening Attendance (Jour/Soir)
| Code | Valeur |
|------|--------|
| 1 | Daytime (Jour) |
| 0 | Evening (Soir) |

### Previous Qualification (Qualification antérieure)
| Code | Valeur |
|------|--------|
| 1 | Secondary education (Enseignement secondaire) |
| 2 | Higher education - bachelor's degree |
| 3 | Higher education - degree |
| 4 | Higher education - master's |
| 5 | Higher education - doctorate |
| 6 | Frequency of higher education |
| 9 | 12th year of schooling - not completed |
| 10 | 11th year of schooling - not completed |
| 12 | Other - 11th year of schooling |
| 14 | 10th year of schooling |
| 15 | 10th year of schooling - not completed |
| 19 | Basic education 3rd cycle (9th/10th/11th year) |
| 38 | Basic education 2nd cycle (6th/7th/8th year) |
| 39 | Technological specialization course |
| 40 | Higher education - degree (1st cycle) |
| 42 | Professional higher technical course |
| 43 | Higher education - master (2nd cycle) |

### Nacionality (Nationalité)
| Code | Valeur |
|------|--------|
| 1 | Portuguese |
| 2 | German |
| 6 | Spanish |
| 11 | Italian |
| 13 | Dutch |
| 14 | English |
| 17 | Lithuanian |
| 21 | Angolan |
| 22 | Cape Verdean |
| 24 | Guinean |
| 25 | Mozambican |
| 26 | Santomean |
| 32 | Turkish |
| 41 | Brazilian |
| 62 | Romanian |
| 100 | Moldova (Republic of) |
| 101 | Mexican |
| 103 | Ukrainian |
| 105 | Russian |
| 108 | Cuban |
| 109 | Colombian |

### Mother's/Father's Qualification (Qualification des parents)
| Code | Valeur |
|------|--------|
| 1 | Secondary Education - 12th Year or Eq. |
| 2 | Higher Education - Bachelor's Degree |
| 3 | Higher Education - Degree |
| 4 | Higher Education - Master's |
| 5 | Higher Education - Doctorate |
| 6 | Frequency of Higher Education |
| 9 | 12th Year - Not Completed |
| 10 | 11th Year - Not Completed |
| 11 | 7th Year (Old) |
| 12 | Other - 11th Year |
| 14 | 10th Year |
| 18 | General commerce course |
| 19 | Basic Education 3rd Cycle |
| 22 | Technical-professional course |
| 26 | 7th year of schooling |
| 27 | 2nd cycle of the general high school |
| 29 | 9th Year - Not Completed |
| 30 | 8th year of schooling |
| 34 | Unknown |
| 35 | Can't read or write |
| 36 | Can read without having a 4th year |
| 37 | Basic education 1st cycle (4th year) |
| 38 | Basic Education 2nd Cycle |
| 39 | Technological specialization course |
| 40 | Higher education - degree (1st cycle) |
| 41 | Specialized higher studies course |
| 42 | Professional higher technical course |
| 43 | Higher Education - Master (2nd cycle) |
| 44 | Higher Education - Doctorate (3rd cycle) |

### Mother's/Father's Occupation (Profession des parents)
| Code | Valeur |
|------|--------|
| 0 | Student |
| 1 | Representatives of the Legislative Power and Executive Bodies |
| 2 | Specialists in Intellectual and Scientific Activities |
| 3 | Intermediate Level Technicians and Professions |
| 4 | Administrative staff |
| 5 | Personal Services, Security and Safety Workers |
| 6 | Farmers and Skilled Workers in Agriculture |
| 7 | Skilled Workers in Industry, Construction |
| 8 | Installation and Machine Operators |
| 9 | Unskilled Workers |
| 10 | Armed Forces Professions |
| 90 | Other Situation |
| 99 | (blank) |
| 122 | Health professionals |
| 123 | Teachers |
| 125 | ICT specialists |
| 131 | Science and engineering technicians |
| 132 | Health associate professionals |
| 134 | Legal, social, sports professionals |
| 141 | Office workers, secretaries |
| 143 | Data, accounting workers |
| 144 | Other administrative support |
| 151 | Personal service workers |
| 152 | Sellers |
| 153 | Personal care workers |
| 171 | Construction workers |
| 173 | Printing, precision workers |
| 175 | Food processing workers |
| 191 | Cleaning workers |
| 192 | Unskilled agriculture workers |
| 193 | Unskilled industry workers |
| 194 | Meal preparation assistants |
| 195 | Street vendors |

---

## Variables Binaires (0/1)

| Variable | 0 | 1 |
|----------|---|---|
| Displaced | Non | Oui (déplacé de résidence) |
| Educational special needs | Non | Oui (besoins éducatifs spéciaux) |
| Debtor | Non | Oui (débiteur) |
| Tuition fees up to date | Non | Oui (frais à jour) |
| Gender | Female | Male |
| Scholarship holder | Non | Oui (boursier) |
| International | Non | Oui (étudiant international) |

---

## Variables Numériques

### Admission
| Variable | Description | Plage |
|----------|-------------|-------|
| Previous qualification (grade) | Note de la qualification antérieure | 0-200 |
| Admission grade | Note d'admission | 0-200 |
| Age at enrollment | Âge à l'inscription | Variable |

### Performance Académique - 1er Semestre
| Variable | Description |
|----------|-------------|
| Curricular units 1st sem (credited) | Unités créditées |
| Curricular units 1st sem (enrolled) | Unités inscrites |
| Curricular units 1st sem (evaluations) | Unités évaluées |
| Curricular units 1st sem (approved) | Unités validées |
| Curricular units 1st sem (grade) | Note moyenne (0-20) |
| Curricular units 1st sem (without evaluations) | Unités sans évaluation |

### Performance Académique - 2ème Semestre
| Variable | Description |
|----------|-------------|
| Curricular units 2nd sem (credited) | Unités créditées |
| Curricular units 2nd sem (enrolled) | Unités inscrites |
| Curricular units 2nd sem (evaluations) | Unités évaluées |
| Curricular units 2nd sem (approved) | Unités validées |
| Curricular units 2nd sem (grade) | Note moyenne (0-20) |
| Curricular units 2nd sem (without evaluations) | Unités sans évaluation |

### Indicateurs Macroéconomiques
| Variable | Description |
|----------|-------------|
| Unemployment rate | Taux de chômage (%) |
| Inflation rate | Taux d'inflation (%) |
| GDP | PIB |

---

## Variable Cible

### Target
| Valeur | Description |
|--------|-------------|
| Dropout | Abandon des études |
| Enrolled | Toujours inscrit |
| Graduate | Diplômé |

---
