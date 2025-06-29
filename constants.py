# STUDENTS GENERATOR
STUDENTS_COUNT = 115
MALE = "M"
FEMALE = "F"
ID = "id"
GENDER = "gender"
CLASS_ID = "class_id"
FIRST_NAME = "first_name"
LAST_NAME = "last_name"
BIRTH_DATE = "birt_date"
AGE = "age"
DEFERRAL = "deferral"
TOGETHER = "together"
TOGETHER_01 = "together_01"
TOGETHER_02 = "together_02"
TOGETHER_03 = "together_03"
TOGETHER_04 = "together_04"
TOGETHER_05 = "together_05"
TOGETHER_06 = "together_06"
TOGETHER_07 = "together_07"
LEARNING_DISABILITIES = "learning_disabilities"
TALENT = "talent"
DIFF_MOTHER_LANG = "diff_mother_lang"
NOT_TOGETHER_01 = "not_together_01"
NOT_TOGETHER_02 = "not_together_02"
NOT_TOGETHER_03 = "not_together_03"
NOT_TOGETHER_04 = "not_together_04"
NOT_TOGETHER_05 = "not_together_05"
NOT_TOGETHER_06 = "not_together_06"
NOT_TOGETHER_07 = "not_together_07"
STUDENTS_FILENAME = "input_data/students_basic.xlsx"


# COMMON
NUM_CLASSES = 4
# NUM_CLASSES = 8
HALL_OF_FAME_SIZE = 5
STUDENTS_PATH = "input_data/students_basic.xlsx"
BEAM_SEARCH = "Paprskové prohledávání"
SIMULATED_ANNEALING = "Simulované žíhání"
EA_DEAP = "Evoluční algoritmus (DEAP)"
EA_OWN = "Evoluční algoritmus (vlastní)"

# SIMULATED ANNEALING
INITIAL_TEMP = 750
COOLING_RATE = 0.998
MAX_ITERATIONS = 100
# MAX_ITERATIONS = 10000

# BEAM SEARCH
BEAM_WIDTH = 5
BEAM_ITERATIONS = 100
# BEAM_ITERATIONS = 600

# EVOLUTIONARY ALGORITHM
POPULATION_SIZE = 100
# GENERATIONS = 1000
GENERATIONS = 100
CX_PROB = 0.8  # Crossover probability
MUT_PROB = 0.2  # Mutation probability
TOURNAMENT_SIZE = 3  # Tournament size
ELITE_COUNT = 2  # Number of elite individuals to carry over to the next generation
