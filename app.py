import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## Function To get response from the model

def getResponse(input_text,no_words,blog_style):

    ### LLama2 model
    # llm=CTransformers(model='model/llama-2-7b.Q5_K_M.gguf',
    #                   model_type='llama',
    #                   config={'max_new_tokens':256,
    #                           'temperature':0.3})
    
    # Yarn Mistral Model
    config = {'max_new_tokens': 512, 'repetition_penalty': 1.1, 'temperature':0.7, 'gpu_layers':0}
    llm = CTransformers(model='model/yarn-mistral-7b-64k.Q5_K_M.gguf', 
                    model_type='mistral', 
                    config=config
                    )
    
    ## Prompt Template

    template="""Create a blog post on {input_text} in MAXIMUM {no_words} words for {blog_style}. Separate each paragraph by 2 lines.
                Output:"""
    # template="""Create a small writeup on {input_text} in about {no_words} words for  {blog_style}.
    #         """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    print(prompt)
    
    ## Generate the ressponse from the model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                    page_icon='💡',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Create Short Blogs 💡") 

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientists','Average person', '5 year old'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getResponse(input_text,no_words,blog_style))