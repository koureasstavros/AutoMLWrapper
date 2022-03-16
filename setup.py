from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
  name = 'automlwrapper',          # How you named your package folder (MyLib)
  packages = find_packages('src'), # Chose the same as "name"
  package_dir={'': 'src'},
  version = '0.4.4',      # Start with a small number and increase it with every change you make
  license = 'MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Automatic Machine Learning Wrapper Library',   # Give a short description about your library
  long_description = long_description,                          # Give a long description about your library
  long_description_content_type = 'text/markdown',
  author = 'Stavros Koureas',                   # Type in your name
  author_email = 'koureasstavros@gmail.com',    # Type in your E-Mail
  url = 'https://github.com/koureasstavros/AutoMLWrapper',   # Provide either the link to your github or to your website
  #download_url = 'https://github.com/koureasstavros/AutoMLWrapper/archive/v0.0.1.tar.gz',    # I explain this later on
  project_urls = {
  'Colab Notebook': 'https://colab.research.google.com/drive/1isX1RB5EeOxFARZ2J-Ekfjbo-PKMTnwi?usp=sharing'
  },
  keywords = ['Machine', 'Learning', 'Automatic'],   # Keywords that define your package best
  python_requires = '>=3.7',
  install_requires = [
          'sklearn',
          'imblearn',
          'scipy',
          'scikit-optimize>=0.9.0'
      ],
  classifiers = [
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.7',    #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.8',    #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.9',    #Specify which pyhton versions that you want to support
  ],
)
