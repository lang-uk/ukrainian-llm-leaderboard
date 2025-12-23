#!/usr/bin/env python3
import re
import os

def camel_to_snake(name):
    """Convert CamelCase to snake_case"""
    # Remove MMLU prefix and Task suffix
    name = name.replace('MMLU', '').replace('Task', '')
    
    # Insert underscores before uppercase letters (except the first one)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# List of all MMLU task classes
classes = [
    'MMLUElectricalEngineeringTask',
    'MMLUMarketingTask',
    'MMLUComputerSecurityTask',
    'MMLUMoralDisputesTask',
    'MMLUHighSchoolMathematicsTask',
    'MMLUVirologyTask',
    'MMLUCollegeComputerScienceTask',
    'MMLUMoralScenariosTask',
    'MMLUJurisprudenceTask',
    'MMLUHighSchoolChemistryTask',
    'MMLUUsForeignPolicyTask',
    'MMLUMedicalGeneticsTask',
    'MMLUCollegeMathematicsTask',
    'MMLUCollegeMedicineTask',
    'MMLUHighSchoolUsHistoryTask',
    'MMLUMachineLearningTask',
    'MMLUProfessionalAccountingTask',
    'MMLUHighSchoolGovernmentAndPoliticsTask',
    'MMLUHighSchoolMicroeconomicsTask',
    'MMLUConceptualPhysicsTask',
    'MMLUHighSchoolMacroeconomicsTask',
    'MMLUHighSchoolWorldHistoryTask',
    'MMLUPhilosophyTask',
    'MMLUSociologyTask',
    'MMLUHighSchoolGeographyTask',
    'MMLUEconometricsTask',
    'MMLUNutritionTask',
    'MMLUWorldReligionsTask',
    'MMLUHighSchoolComputerScienceTask',
    'MMLUBusinessEthicsTask',
    'MMLUGlobalFactsTask',
    'MMLUAnatomyTask',
    'MMLUHighSchoolEuropeanHistoryTask',
    'MMLUProfessionalLawTask',
    'MMLUManagementTask',
    'MMLUHighSchoolBiologyTask',
    'MMLUPublicRelationsTask',
    'MMLUAstronomyTask',
    'MMLUProfessionalMedicineTask',
    'MMLUHighSchoolPsychologyTask',
    'MMLUCollegeChemistryTask',
    'MMLUHumanSexualityTask',
    'MMLULogicalFallaciesTask',
    'MMLUMiscellaneousTask',
    'MMLUHighSchoolStatisticsTask',
    'MMLUProfessionalPsychologyTask',
    'MMLUHumanAgingTask',
    'MMLUElementaryMathematicsTask',
    'MMLUFormalLogicTask',
    'MMLUCollegePhysicsTask',
    'MMLUHighSchoolPhysicsTask',
    'MMLUAbstractAlgebraTask',
    'MMLUPrehistoryTask',
    'MMLUClinicalKnowledgeTask',
    'MMLUInternationalLawTask',
    'MMLUCollegeBiologyTask',
    'MMLUSecurityStudiesTask'
]

# Create directory if it doesn't exist
output_dir = 'tasks/ukrainian_bench/mmlu_uk'
os.makedirs(output_dir, exist_ok=True)

# Generate YAML files
for class_name in classes:
    task_name = f"mmlu_uk_{camel_to_snake(class_name)}"
    yaml_filename = f"{task_name}.yaml"
    yaml_path = os.path.join(output_dir, yaml_filename)
    
    yaml_content = f"""task: {task_name}
class: !function mmlu_tasks.{class_name}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created: {yaml_path}")

print(f"\nGenerated {len(classes)} YAML files in {output_dir}") 