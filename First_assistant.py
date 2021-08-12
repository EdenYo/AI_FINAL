import Gmm
import os
if __name__ == '__main__':
    csv_file_path = os.path.join(os.path.dirname(__file__), 'Data', 'mushrooms_data.txt')
    Gmm.main(csv_file_path, False)