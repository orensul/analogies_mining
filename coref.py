import os, stat

coref_predict_file_path = 'predict.py'
coref_input_file_path = 'input_files/input.jsonl'

def main():
    os.chmod(coref_predict_file_path, stat.S_IRWXU)
    command = "python " + coref_predict_file_path + " --input_file " + coref_input_file_path
    os.system(command)



if __name__ == '__main__':
    main()

