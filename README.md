# NADRA_FR
implemented-spoofing-and embeding model 

---

requirement:  python 3.9 


pip install -r requirements.txt 


---

place spoofing and embeding models in the same folder 


best_finetuned.pth 
final_model.pth
model.onnx


---



in terminal 1 run 

uvicorn api_server:app --reload --port 8000


terminal 2 run  

streamlit run streamlit_frontend.py


---

app available on 
  Local URL: http://localhost:8501
  
  Network URL: http://192.168.1.14:8501
