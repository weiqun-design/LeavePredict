import copy
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder


features = ['Age', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently', 'DistanceFromHome', 'education_1',
                'education_2','education_3', 'education_4', 'education_5', 'EnvironmentSatisfaction',
                'Gender', 'JobInvolvement', 'JobLevel_1','JobLevel_2', 'JobLevel_3','JobLevel_4',"JobLevel_5",
                'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
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


def mapping_data(data):
    train_data = copy.deepcopy(data)
    # 将离职结果mapping为0和1
    attrition_mapping = {"No": 0, "Yes": 1}
    train_data['Attrition'] = train_data['Attrition'].map(attrition_mapping)

    # gender mapping为 "Male": 1, "Female": 0
    gender_mapping = {"Male": 1, "Female": 0}
    train_data['Gender'] = train_data['Gender'].map(gender_mapping)

    # OverTime mapping为 "Yes": 1, "No": 0
    over_time_mapping = {"Yes": 1, "No": 0}
    train_data['OverTime'] = train_data['OverTime'].map(over_time_mapping)
    return train_data


def deal_with_business_travel(data, origin_index):
    train_data = copy.deepcopy(data)
    business_non_travel = [0 for i in range(train_data.shape[0])]
    business_travel_rarely = [0 for i in range(train_data.shape[0])]
    business_travel_frequently = [0 for i in range(train_data.shape[0])]
    for i in range(train_data.shape[0]):
        if train_data['BusinessTravel'].get(i + origin_index) == 'Non-Travel':
            business_non_travel[i] = 1
            continue
        if train_data['BusinessTravel'].get(i + origin_index) == 'Travel_Rarely':
            business_travel_rarely[i] = 1
            continue
        if train_data['BusinessTravel'].get(i + origin_index) == 'Travel_Frequently':
            business_travel_frequently[i] = 1
            continue
        print(str(train_data[i:i + 1]['BusinessTravel'].get(0)) + ' error')
    train_data['Non-Travel'] = business_non_travel
    train_data['Travel_Rarely'] = business_travel_rarely
    train_data['Travel_Frequently'] = business_travel_frequently
    return train_data


def deal_with_Education(data, origin_index):
    train_data = copy.deepcopy(data)
    education_1 = [0 for i in range(train_data.shape[0])]
    education_2 = [0 for i in range(train_data.shape[0])]
    education_3 = [0 for i in range(train_data.shape[0])]
    education_4 = [0 for i in range(train_data.shape[0])]
    education_5 = [0 for i in range(train_data.shape[0])]
    for i in range(train_data.shape[0]):
        if train_data['Education'].get(i + origin_index) == 1:
            education_1[i] = 1
            continue
        if train_data['Education'].get(i + origin_index) == 2:
            education_2[i] = 1
            continue
        if train_data['Education'].get(i + origin_index) == 3:
            education_3[i] = 1
            continue
        if train_data['Education'].get(i + origin_index) == 4:
            education_4[i] = 1
            continue
        if train_data['Education'].get(i + origin_index) == 5:
            education_5[i] = 1
            continue
        print(str(train_data[i:i + 1]['Education'].get(0)) + ' error')
    train_data['education_1'] = education_1
    train_data['education_2'] = education_2
    train_data['education_3'] = education_3
    train_data['education_4'] = education_4
    train_data['education_5'] = education_5
    return train_data

def deal_with_job_level(data, origin_index):
    train_data = copy.deepcopy(data)
    JobLevel_1 = [0 for i in range(train_data.shape[0])]
    JobLevel_2 = [0 for i in range(train_data.shape[0])]
    JobLevel_3 = [0 for i in range(train_data.shape[0])]
    JobLevel_4 = [0 for i in range(train_data.shape[0])]
    JobLevel_5 = [0 for i in range(train_data.shape[0])]
    for i in range(train_data.shape[0]):
        if train_data['JobLevel'].get(i + origin_index) == 1:
            JobLevel_1[i] = 1
            continue
        if train_data['JobLevel'].get(i + origin_index) == 2:
            JobLevel_2[i] = 1
            continue
        if train_data['JobLevel'].get(i + origin_index) == 3:
            JobLevel_3[i] = 1
            continue
        if train_data['JobLevel'].get(i + origin_index) == 4:
            JobLevel_4[i] = 1
            continue
        if train_data['JobLevel'].get(i + origin_index) == 5:
            JobLevel_5[i] = 1
            continue
        print(str(train_data[i:i + 1]['JobLevel'].get(0)) + ' error')
    train_data['JobLevel_1'] = JobLevel_1
    train_data['JobLevel_2'] = JobLevel_2
    train_data['JobLevel_3'] = JobLevel_3
    train_data['JobLevel_4'] = JobLevel_4
    train_data['JobLevel_5'] = JobLevel_5
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
    # print("train feature before scale" + str(train_data))
    train_data = scale_train_data(train_data)
    # print("train feature after scale" + str(train_data))
    train_data = deal_with_business_travel(train_data,train_origin_index)
    train_data = deal_with_Education(train_data,train_origin_index)
    train_data = deal_with_department(train_data,train_origin_index)
    train_data = deal_with_education_field(train_data,train_origin_index)
    train_data = deal_with_marital_status(train_data,train_origin_index)
    train_data = deal_with_job_role(train_data,train_origin_index)
    train_data = deal_with_job_level(train_data,train_origin_index)
    train_data = mapping_data(train_data)
    dict_vectorizer = DictVectorizer(sparse=False)
    train_feature = train_data[features]
    train_feature = dict_vectorizer.fit_transform(train_feature.to_dict(orient='record'))
    train_labels = train_data['Attrition']

    print("prepare for test data")
    test_data = scale_train_data(test_data)
    test_data = deal_with_business_travel(test_data, test_origin_index)
    test_data = deal_with_Education(test_data, test_origin_index)
    test_data = deal_with_department(test_data,test_origin_index)
    test_data = deal_with_education_field(test_data,test_origin_index)
    test_data = deal_with_marital_status(test_data,test_origin_index)
    test_data = deal_with_job_role(test_data,test_origin_index)
    test_data = deal_with_job_level(test_data, test_origin_index)
    test_data = mapping_data(test_data)
    test_feature = test_data[features]
    test_feature = dict_vectorizer.fit_transform(test_feature.to_dict(orient='record'))
    test_labels = test_data['Attrition']
    test_labels = convert_test_labels(test_labels)

    return train_feature, train_labels, test_feature, test_labels


def prepare_train_data_for_ffm(train_data,test_data,train_origin_index, test_origin_index):
    train_feature, train_labels, test_feature, test_labels = get_train_feature_and_train_labels(train_data,
                                                                                                test_data,
                                                                                                train_origin_index,
                                                                                                test_origin_index)

    field_dict = dict()
    for feature in features:
        if feature == 'Age':
            field_dict[feature] = 0
        elif feature in ['Non-Travel','Travel_Rarely','Travel_Frequently']:
            field_dict[feature] = 1
        elif feature in ['DistanceFromHome']:
            field_dict[feature] = 2
        elif feature in ['education_1', 'education_2', 'education_3', 'education_4' ,'education_5']:
            field_dict[feature] = 3
        elif feature in ['EnvironmentSatisfaction']:
            field_dict[feature] = 4
        elif feature in ['Gender']:
            field_dict[feature] = 5
        elif feature in ['JobInvolvement']:
            field_dict[feature] = 6
        elif feature in ['JobLevel_1','JobLevel_2','JobLevel_3','JobLevel_4','JobLevel_5']:
            field_dict[feature] = 7
        left_list = ['JobSatisfaction','MonthlyIncome','NumCompaniesWorked','OverTime','PercentSalaryHike',
                         'PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
                         'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                         'YearsInCurrentRole','YearsSinceLastPromotion', 'YearsWithCurrManager']
        if feature in left_list:
            for i in range(len(left_list)):
                if left_list[i] == feature:
                    field_dict[feature] = 8 + i

        if feature in ['Department_sales','Department_research','Department_hr']:
            field_dict[feature] = 8 + len(left_list)
        elif feature in ['EducationField_life_sciences','EducationField_technical_degree', 'EducationField_marketing',
                         'EducationField_medical', 'EducationField_hr', 'EducationField_other']:
            field_dict[feature] = 9 + len(left_list)
        elif feature in ['MaritalStatus_married', 'MaritalStatus_single', 'MaritalStatus_divorced']:
            field_dict[feature] = 10 + len(left_list)
        elif feature in ['JobRole_manager', 'JobRole_research_scientist', 'JobRole_sales_executive','JobRole_sales_representative',
                         'JobRole_laboratory_technician', 'JobRole_manufacturing_director', 'JobRole_human_resources',
                         'JobRole_research_director','JobRole_healthcare_representative']:
            field_dict[feature] = 11 + len(left_list)
    print(train_feature.shape)
    print(len(features))
    print(train_feature)
    with open('ffm_train_data.txt','w') as f:
        for j in range(train_feature.shape[0]):
            f.write(str(train_labels[j]))
            f.write(' ')
            for i in range(len(features)):
                if train_feature[j][i] != 0:
                    f.write(str(i))
                    f.write(':')
                    f.write(str(field_dict[features[i]]))
                    f.write(":")
                    f.write(str(train_feature[j][i]))
                    f.write(' ')
            f.write('\n')

    with open('ffm_test_data.txt','w') as f:
        for j in range(test_feature.shape[0]):
            f.write(str(test_labels[j]))
            f.write(' ')
            for i in range(len(features)):
                if test_feature[j][i] != 0:
                    f.write(str(i))
                    f.write(':')
                    f.write(str(field_dict[features[i]]))
                    f.write(":")
                    f.write(str(test_feature[j][i]))
                    f.write(' ')
            f.write('\n')





