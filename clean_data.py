import pandas as pd

def clean_data(file_obj, final_file_name):

    """
    Read a csv file
    """
    data_frame = pd.read_csv(file_obj)
    data_frame.dropna(axis=0, inplace=True)
    data_frame.to_csv(final_file_name)
    return
# if __name__ == "__main__":
#     answers = "/Users/idoilani/Documents/dev/Library-Science/rquestions/Answers.csv"
#     questions = "/Users/idoilani/Documents/dev/Library-Science/rquestions/Questions.csv"
#     tags = "/Users/idoilani/Documents/dev/Library-Science/rquestions/Tags.csv"
#     with open(answers, "rb") as f_obj:
#             clean_data(f_obj, '/Users/idoilani/Documents/dev/Library-Science/rquestions/answers_final.csv')
#     with open(questions, "rb") as f_obj:
#             clean_data(f_obj, '/Users/idoilani/Documents/dev/Library-Science/rquestions/questions_final.csv')
#     with open(tags, "rb") as f_obj:
#             clean_data(f_obj, '/Users/idoilani/Documents/dev/Library-Science/rquestions/tags_final.csv')



