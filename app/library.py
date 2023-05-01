from bs4 import BeautifulSoup
import urllib.request
import ebooklib
from ebooklib import epub
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
import tiktoken
from sys import meta_path
from langchain import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


class Library:

    def __init__(self, books_path=r'./books/') -> None:
        self.books_path = books_path
        self.books = None
        self.book_chunks = None
        self.tiktokenizer = None
        self.embeddings = None
        self.faiss = None
    
    def from_vectorstore(self, local_faiss_store='faiss.vectorstore'):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
        self.faiss = FAISS.load_local(local_faiss_store, embeddings=self.embeddings)
    
    def create_index(self):
        self.__read_books()
        self.__split_book_data()
        self.__embed_book_chunks()

    def __read_books(self):
        # folder path
        # dir_path = r'./books/'
        dir_path = self.books_path

        # list to store files
        epub_files_list = []
        pdf_files_list = []
        # Iterate directory
        for file in os.listdir(dir_path):
            # check only text files
            if file.endswith('.epub'):
                epub_files_list.append(file)
            if file.endswith('.pdf'):
                pdf_files_list.append(file)

        print(epub_files_list)
        print(pdf_files_list)

        data = list()
        data.append(self.__parse_epub_data(epub_files_list))
        data.append(self.__parse_pdf_data(pdf_files_list))

        self.books = data

    # define a length function
    def __tiktoken_len(self, text: str) -> int:
        if self.tiktokenizer is None:
            self.tiktokenizer = tiktoken.get_encoding('cl100k_base')  # cl100k base is encoder used by ada-002
        tokens = self.tiktokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def __split_book_data(self):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,  # number of tokens overlap between chunks
            length_function=self.__tiktoken_len,  # token count function
            separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
        )

        new_data = []

        for row in tqdm(self.books):
            chunks = text_splitter.split_text(row['content'])
            row.pop('content')
            for i, text in enumerate(chunks):
                new_data.append({**row, **{'chunk': i, 'text': text}})
        
        self.book_chunks = new_data
    
    def __embed_book_chunks(self):

        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

        texts_to_embed = list()
        metadata_list = list()

        for _d in self.book_chunks:
            texts_to_embed.append(_d['text'])
            metadata_list.append({k:v for k,v in _d.items() if 'text' not in k})

        self.faiss = FAISS.from_texts(texts_to_embed, self.embeddings, metadatas=metadata_list)
        self.faiss.save_local('faiss.vectorstore')


def chapter_to_str(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p')]
    return ' \n '.join(text)


def remove_xa0(string):
    return string.replace(u'\xa0', u' ')


def parse_epub_data(files_list):
    # create variable for collecting parsed data
    parsed_data = list()

    # loop through all the books
    # parse each one to one data element with all the text of the book
    for book_file in files_list:

        # read file
        book = epub.read_epub(book_file)

        # try to extract title of the epub book
        try:
            title = book.get_metadata('DC', 'title')[0][0]
        except e:
            # if it fails, the title will be the name of the file
            title = book_file

        # create virable to save all the text content of the book
        full_book_text = ''

        # loop through all text content blocks of the book and save them
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            item_name = item.get_name()

            item_content = remove_xa0(chapter_to_str(item))

            full_book_text += item_content

        # create data element with all the content from this book
        book_data = {'source': book_file, 'title': title, 'content': full_book_text}
        
        # save data element
        parsed_data.append(book_data)

    return parsed_data


def parse_pdf_data(files_list):

    parsed_data = list()

    for pdf_file in files_list:
        # creating a pdf reader object
        reader = PdfReader(pdf_file)

        # try to extract title of the file
        info = reader.metadata
        try:
            title = info.title
        except e:
            # if it fails, saved title is the name of the file
            title = pdf_file
        
        full_book_text = ''
            
        # looping through pages
        for page_num in range(len(reader.pages)):
            
            # getting a specific page from the pdf file
            page = reader.pages[page_num]
            
            # extracting text from page
            page_text = page.extract_text()

            # save page text to full book text
            full_book_text += page_text
        
        # save book content
        book_data = {'source': pdf_file, 'title': title, 'content': full_book_text}

        # integrate book content into parsed data
        parsed_data.append(book_data)

    return parsed_data