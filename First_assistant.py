import Gmm
import os
if __name__ == '__main__':
    # read the data
    csv_file_path = os.path.join(os.path.dirname(__file__), 'Data', 'mushrooms_data.txt')
    # run gmm 
    Gmm.main(csv_file_path, False)
