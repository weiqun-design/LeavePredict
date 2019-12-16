import copy
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.preprocessing import scale


def mapping_data(data):
    train_data = copy.deepcopy(data)
    # 将离职结果mapping为0和1
    attrition_mapping = {"No": 0, "Yes": 1}
    train_data['Attrition'] = train_data['Attrition'].map(attrition_mapping)

    # 将businessTravel mapping为 "Non-Travel": 0, "TravelRarely": 1, "TravelFrequently": 2
    business_travel_mapping = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    train_data['BusinessTravel'] = train_data['BusinessTravel'].map(business_travel_mapping)

    # gender mapping为 "Male": 1, "Female": 0
    gender_mapping = {"Male": 1, "Female": 0}
    train_data['Gender'] = train_data['Gender'].map(gender_mapping)

    # OverTime mapping为 "Yes": 1, "No": 0
    over_time_mapping = {"Yes": 1, "No": 0}
    train_data['OverTime'] = train_data['OverTime'].map(over_time_mapping)
    return train_data


# 将Department 拆分后进行one hot
def deal_with_department(data,origin_index):
    train_data = copy.deepcopy(data)
    sales_list = [0 for i in range(train_data.shape[0])]
    research_list = [0 for i in range(train_data.shape[0])]
    hr_list = [0 for i in range(train_data.shape[0])]
    for i in range(train_data.shape[0]):
        if train_data['Department'].get(i+origin_index) == 'Sales':
            sales_list[i] = 1
            continue
        if train_data['Department'].get(i+origin_index) == 'Research & Development':
            research_list[i] = 1
            continue
        if train_data['Department'].get(i+origin_index) == 'Human Resources':
            hr_list[i] = 1
            continue
        print(str(train_data[i:i + 1]['Department'].get(0)) + ' error')
    train_data['Department_sales'] = sales_list
    train_data['Department_research'] = research_list
    train_data['Department_hr'] = hr_list
    return train_data


def deal_with_education_field(data,origin_index):
    train_data = copy.deepcopy(data)
    size_x = train_data.shape[0]
    life_sciences_list = [0 for i in range(size_x)]
    technical_degree_list = [0 for i in range(size_x)]
    marketing_list = [0 for i in range(size_x)]
    medical_list = [0 for i in range(size_x)]
    hr_list = [0 for i in range(size_x)]
    other_list = [0 for i in range(size_x)]
    for i in range(size_x):
        if train_data['EducationField'].get(i+origin_index) == 'Life Sciences':
            life_sciences_list[i] = 1
            continue
        if train_data['EducationField'].get(i+origin_index) == 'Technical Degree':
            technical_degree_list[i] = 1
            continue
        if train_data['EducationField'].get(i+origin_index) == 'Marketing':
            marketing_list[i] = 1
            continue
        if train_data['EducationField'].get(i+origin_index) == 'Medical':
            medical_list[i] = 1
            continue
        if train_data['EducationField'].get(i+origin_index) == 'Human Resources':
            hr_list[i] = 1
            continue
        if train_data['EducationField'].get(i+origin_index) == 'Other':
            other_list[i] = 1
            continue
        print(train_data['EducationField'].get(i+origin_index) + ' Error')
    train_data['EducationField_life_sciences'] = life_sciences_list
    train_data['EducationField_technical_degree'] = technical_degree_list
    train_data['EducationField_marketing'] = marketing_list
    train_data['EducationField_medical'] = medical_list
    train_data['EducationField_hr'] = hr_list
    train_data['EducationField_other'] = other_list
    return train_data


def deal_with_marital_status(data,origin_index):
    train_data = copy.deepcopy(data)
    size_x = train_data.shape[0]
    married_list = [0 for i in range(size_x)]
    single_list = [0 for i in range(size_x)]
    divorced_list = [0 for i in range(size_x)]
    for i in range(size_x):
        if train_data['MaritalStatus'].get(i+origin_index) == 'Married':
            married_list[i] = 1
            continue
        elif train_data['MaritalStatus'].get(i+origin_index) == 'Single':
            single_list[i] = 1
            continue
        elif train_data['MaritalStatus'].get(i+origin_index) == 'Divorced':
            divorced_list[i] = 1
            continue
        else:
            print(train_data['MaritalStatus'] + " error")
    train_data['MaritalStatus_married'] = married_list
    train_data['MaritalStatus_single'] = single_list
    train_data['MaritalStatus_divorced'] = divorced_list
    return train_data


# deal_with_job_role
def deal_with_job_role(data,origin_index):
    train_data = copy.deepcopy(data)
    size_x = train_data.shape[0]
    manager_list = [0 for i in range(size_x)]
    research_scientist_list = [0 for i in range(size_x)]
    sales_executive_list = [0 for i in range(size_x)]
    sales_representative_list = [0 for i in range(size_x)]
    laboratory_technician_list = [0 for i in range(size_x)]
    manufacturing_director_list = [0 for i in range(size_x)]
    hr_list = [0 for i in range(size_x)]
    research_director_list = [0 for i in range(size_x)]
    healthcare_representative_list = [0 for i in range(size_x)]
    unique_value = ['Manager', 'Research Scientist', 'Sales Executive', 'Sales Representative', 'Laboratory Technician',
                    'Manufacturing Director', 'Human Resources', 'Research Director', 'Healthcare Representative']
    role_list = [manager_list, research_scientist_list, sales_executive_list, sales_representative_list,
                 laboratory_technician_list,
                 manufacturing_director_list, hr_list, research_director_list, healthcare_representative_list]
    for i in range(size_x):
        modified = False
        for j in range(len(unique_value)):
            if train_data['JobRole'].get(i+origin_index) == unique_value[j]:
                modified = True
                role_list[j][i] = 1
                break
        if not modified:
            print(train_data['JobRole'].get(i+origin_index) + " error")
    for j in range(len(role_list)):
        train_data["JobRole_" + unique_value[j].replace(' ', '_').lower()] = role_list[j]
    return train_data


def convert_test_labels(test_labels):
    copy_array = list()
    for data in test_labels:
        copy_array.append(data)
    test_labels_new = pd.Series(copy_array)
    return test_labels_new


def scale_train_data(train_data):
    to_be_scale_list = ['Age','DistanceFromHome','MonthlyIncome','PercentSalaryHike']
    for i in range(len(to_be_scale_list)):
        train_data[to_be_scale_list[i]] = scale(train_data[to_be_scale_list[i]])
    return train_data


def get_train_feature_and_train_labels(train_data,test_data,train_origin_index, test_origin_index):
    print("train feature before scale" + str(train_data))
    train_data = scale_train_data(train_data)
    print("train feature after scale" + str(train_data))
    train_data = deal_with_department(train_data,train_origin_index)
    train_data = deal_with_education_field(train_data,train_origin_index)
    train_data = deal_with_marital_status(train_data,train_origin_index)
    train_data = deal_with_job_role(train_data,train_origin_index)
    train_data = mapping_data(train_data)
    dict_vectorizer = DictVectorizer(sparse=False)
    features = ['Age', 'BusinessTravel', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                'Gender', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Department_sales', 'Department_research',
                'Department_hr',
                'EducationField_life_sciences', 'EducationField_technical_degree', 'EducationField_marketing',
                'EducationField_medical',
                'EducationField_hr', 'EducationField_other', 'MaritalStatus_married', 'MaritalStatus_single',
                'MaritalStatus_divorced',
                'JobRole_manager', 'JobRole_research_scientist', 'JobRole_sales_executive',
                'JobRole_sales_representative',
                'JobRole_laboratory_technician', 'JobRole_manufacturing_director', 'JobRole_human_resources',
                'JobRole_research_director',
                'JobRole_healthcare_representative']
    train_feature = train_data[features]
    train_feature = dict_vectorizer.fit_transform(train_feature.to_dict(orient='record'))
    train_labels = train_data['Attrition']

    test_data = scale_train_data(test_data)
    test_data = deal_with_department(test_data,test_origin_index)
    test_data = deal_with_education_field(test_data,test_origin_index)
    test_data = deal_with_marital_status(test_data,test_origin_index)
    test_data = deal_with_job_role(test_data,test_origin_index)
    test_data = mapping_data(test_data)
    test_feature = test_data[features]
    test_feature = dict_vectorizer.fit_transform(test_feature.to_dict(orient='record'))
    test_labels = test_data['Attrition']
    test_labels = convert_test_labels(test_labels)

    return train_feature, train_labels, test_feature, test_labels


# def prepare_test_file(test_feature):
#     with open("test_for_fm.txt",'wb') as f:
#         for i in range(test_feature.shape[0]):
#             for j in range(test_feature.shape[1]):
#                 if j != test_feature.shape[1] - 1:
#                     if int(test_feature[i][j]) != 0:
#                         f.write((str(j) + ':' + str(int(test_feature[i][j])) + " ").encode("utf-8"))
#                 else:
#                     if int(test_feature[i][j]) != 0:
#                         f.write((str(int(test_feature[i][j])) + "\n").encode("utf-8"))
#                     else:
#                         f.write(("\n").encode("utf-8"))






