from setuptools import find_packages,setup 
setup(
    name='mcqgenerator',
    version='0.0.1',
    author='Thoufiq-Developer',
    author_email='thoufiq2005ahamed@gmail.com',
    install_requires=['openai','langchain','streamlit','python-dotenv','PyPDF2'],
    packages=find_packages()
)
