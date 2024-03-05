# This Python GUI application is designed to:
# - Define an App class
# - Define a TabView class

print("\n\nLoading system...\n\n")

import joblib
import numpy as np
import contractions
import pandas as pd
import seaborn as sns
from PIL import Image
import customtkinter as ctk
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from xgboost import plot_importance
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("XGBERT")
        self.geometry("1440x810")
        self.resizable(False, False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        self.tab_view = TabView(master=self, width=1400, height=700)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.appearance_mode = ctk.StringVar(value="on")
        self.appearance_switch = ctk.CTkSwitch(master=self, switch_width=40, text="Dark Mode", font=("Arial", 13), command=self.appearance, variable=self.appearance_mode, onvalue="on", offvalue="off")
        self.appearance_switch.grid(row=1, column=0, padx=20, pady=(0, 40), sticky="ew")

        self.update_idletasks()
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        size = tuple(int(_) for _ in self.geometry().split('+')[0].split('x'))
        x = width/2 - size[0]/2
        y = height/2 - size[1]/2
        self.geometry("%dx%d+%d+%d" % (size + (x, y)))


    def appearance(self):
        if self.appearance_mode.get() == "on":
            ctk.set_appearance_mode("dark")
            #ctk.set_default_color_theme("blue")

        else:
            ctk.set_appearance_mode("light")
            #ctk.set_default_color_theme("green")


class TabView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.add("Sentiment Analysis")  
        self.add("Data Visualization")
        self.add("Model Comparison")  

        self.create_sentiment_analysis_tab()
        self.create_data_visualization_tab()
        self.create_model_comparison_tab()


    def create_sentiment_analysis_tab(self):
        self.tab("Sentiment Analysis").grid_columnconfigure((0, 1), weight=1)
        self.tab("Sentiment Analysis").grid_rowconfigure((0, 1), weight=1)
        
        self.sliders_frame = ctk.CTkFrame(master=self.tab("Sentiment Analysis"), width=400, height=450, corner_radius=20, border_width=2, bg_color="transparent")
        self.sliders_frame.grid(row=0, column=0, columnspan=2, padx=(60, 0), pady=(20, 0), sticky="nw")
        self.sliders_frame.grid_propagate(False)
        self.sliders_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.sliders_frame.grid_rowconfigure((0, 1), weight=1)
        self.sliders_frame_label = ctk.CTkLabel(master=self.sliders_frame, text="Feature Sliders", font=("Arial", 24))
        self.sliders_frame_label.grid(row=0, column=0, columnspan=3, padx=(20, 0), pady=(20, 0), sticky="nw")

        self.upvotes_label = ctk.CTkLabel(master=self.sliders_frame, text="Upvotes", font=("Arial", 13))
        self.upvotes_label.grid(row=0, column=0, padx=(45, 0), pady=(120, 0), sticky="sw")
        self.upvotes_slider = ctk.CTkSlider(master=self.sliders_frame, height=220, from_=0, to=10, number_of_steps=10, orientation="vertical", command=self.update_upvotes_label)
        self.upvotes_slider.grid(row=1, column=0, padx=(60, 0), pady=(0, 90), sticky="sw")
        self.upvotes_value_label = ctk.CTkLabel(master=self.sliders_frame, text="5", font=("Arial", 13))
        self.upvotes_value_label.grid(row=1, column=0, padx=(2, 0), pady=(0, 60), sticky="swe")

        self.total_votes_label = ctk.CTkLabel(master=self.sliders_frame, text="Total Votes", font=("Arial", 13))
        self.total_votes_label.grid(row=0, column=1, padx=(39, 0), pady=(120, 0), sticky="sw")
        self.total_votes_slider = ctk.CTkSlider(master=self.sliders_frame, height=220, from_=0, to=10, number_of_steps=10, orientation="vertical", command=self.update_total_votes_label)
        self.total_votes_slider.grid(row=1, column=1, padx=(64, 0), pady=(0, 90), sticky="sw")
        self.total_votes_value_label = ctk.CTkLabel(master=self.sliders_frame, text="5", font=("Arial", 13))
        self.total_votes_value_label.grid(row=1, column=1, padx=(0, 4), pady=(0, 60), sticky="swe")
        
        self.rating_label = ctk.CTkLabel(master=self.sliders_frame, text="Rating", font=("Arial", 13))
        self.rating_label.grid(row=0, column=2, padx=(38, 0), pady=(120, 0), sticky="sw")
        self.rating_slider = ctk.CTkSlider(master=self.sliders_frame, height=220, from_=1, to=5, number_of_steps=4, orientation="vertical", command=self.update_rating_label)
        self.rating_slider.grid(row=1, column=2, padx=(48, 0), pady=(0, 90), sticky="sw")
        self.rating_value_label = ctk.CTkLabel(master=self.sliders_frame, text="3", font=("Arial", 13))
        self.rating_value_label.grid(row=1, column=2, padx=(0, 6), pady=(0, 60), sticky="swe")

        self.user_review_textbox = ctk.CTkTextbox(master=self.tab("Sentiment Analysis"), width=1200, height=100, corner_radius=20, border_width=2, fg_color="transparent", font=("Arial", 18))
        self.user_review_textbox.grid(row=1, column=0, padx=(60, 0), pady=(20, 40), sticky="sw")
        self.user_review_textbox.insert("0.0", "Enter your review here...")
        self.user_review_textbox.bind("<FocusIn>", lambda event: self.user_review_textbox.delete("0.0", "end"))

        self.predict_button = ctk.CTkButton(master=self.tab("Sentiment Analysis"), width=50, height=50, corner_radius=15, border_width=0, text="↑", font=("Arial", 24), anchor="center", command=predict)
        self.predict_button.grid(row=1, column=1, padx=(0, 60), pady=(0, 65), sticky="se")

        self.results_frame = ctk.CTkFrame(master=self.tab("Sentiment Analysis"), width=820, height=450, corner_radius=20, border_width=2, bg_color="transparent")
        self.results_frame.grid(row=0, column=0, columnspan=2, padx=(0, 60), pady=(20, 0), sticky="ne")
        self.results_frame.grid_propagate(False)
        self.results_frame.grid_columnconfigure((0, 1), weight=1)
        self.results_frame.grid_rowconfigure((0, 1), weight=1)
        self.results_frame_label = ctk.CTkLabel(master=self.results_frame, text="Model Prediction and Evaluation Metrics", font=("Arial", 24))
        self.results_frame_label.grid(row=0, column=0, columnspan=3, padx=(0, 20), pady=(20, 0), sticky="ne")

        self.result = Image.open("images/welcome.png")
        self.result_image = ctk.CTkImage(light_image=self.result, size=(100, 100))
        self.result_image_show = ctk.CTkLabel(master=self.results_frame, image=self.result_image, text="", bg_color="transparent")
        self.result_image_show.grid(row=0, column=0, padx=(140, 0), pady=(120, 0), sticky="nw")
        self.result_text = ctk.CTkLabel(master=self.results_frame, text="Hi there! Input a product review \nand I'll analyze its sentiment.", font=("Arial", 18), bg_color="transparent")
        self.result_text.grid(row=0, column=0, padx=(66, 0), pady=(240, 0), sticky="nw")

        self.accuracy_score_label = ctk.CTkLabel(master=self.results_frame, text="Accuracy", font=("Arial", 13), bg_color="transparent")
        self.accuracy_score_label.grid(row=0, column=1, padx=(0, 374), pady=(120, 0), sticky="ne")
        self.accuracy_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.accuracy_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(150, 0), sticky="ne")
        self.accuracy_score_bar.set(value=0.88)
        self.accuracy_score_percentage = ctk.CTkLabel(master=self.results_frame, text="88%", font=("Arial", 13), bg_color="transparent")
        self.accuracy_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(139, 0), sticky="ne")

        self.precision_score_label = ctk.CTkLabel(master=self.results_frame, text="Precision", font=("Arial", 13), bg_color="transparent")
        self.precision_score_label.grid(row=0, column=1, padx=(0, 376), pady=(180, 0), sticky="ne")
        self.precision_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.precision_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(210, 0), sticky="ne")
        self.precision_score_bar.set(value=0.86)
        self.precision_score_percentage = ctk.CTkLabel(master=self.results_frame, text="86%", font=("Arial", 13), bg_color="transparent")
        self.precision_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(199, 0), sticky="ne")

        self.recall_score_label = ctk.CTkLabel(master=self.results_frame, text="Recall", font=("Arial", 13), bg_color="transparent")
        self.recall_score_label.grid(row=0, column=1, padx=(0, 394), pady=(240, 0), sticky="ne")
        self.recall_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.recall_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(270, 0), sticky="ne")
        self.recall_score_bar.set(value=0.84)
        self.recall_score_percentage = ctk.CTkLabel(master=self.results_frame, text="84%", font=("Arial", 13), bg_color="transparent")
        self.recall_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(259, 0), sticky="ne")

        self.f1_score_label = ctk.CTkLabel(master=self.results_frame, text="F1", font=("Arial", 13), bg_color="transparent")
        self.f1_score_label.grid(row=0, column=1, padx=(0, 414), pady=(300, 0), sticky="ne")
        self.f1_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.f1_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(330, 0), sticky="ne")
        self.f1_score_bar.set(value=0.82)
        self.f1_score_percentage = ctk.CTkLabel(master=self.results_frame, text="82%", font=("Arial", 13), bg_color="transparent")
        self.f1_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(319, 0), sticky="ne")

        self.cv_score_label = ctk.CTkLabel(master=self.results_frame, text="Cross-Validation", font=("Arial", 13), bg_color="transparent")
        self.cv_score_label.grid(row=0, column=1, padx=(0, 335), pady=(360, 0), sticky="ne")
        self.cv_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.cv_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(390, 0), sticky="ne")
        self.cv_score_bar.set(value=0.86)
        self.cv_score_percentage = ctk.CTkLabel(master=self.results_frame, text="86%", font=("Arial", 13), bg_color="transparent")
        self.cv_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(379, 0), sticky="ne")

    
    def update_sentiment_display(self, y_pred):
        self.happy = Image.open("images/happy.png")
        self.neutral = Image.open("images/neutral.png")
        self.sad = Image.open("images/sad.png")

        value = y_pred

        if value == 0:
            self.result_image = ctk.CTkImage(light_image=self.sad, size=(100, 100))
            self.result_text.configure(text="The model predicted the \nreview to be negative.")
            self.result_text.grid_configure(row=0, column=0, padx=(92, 0), pady=(240, 0), sticky="nw")

        elif value == 1:
            self.result_image = ctk.CTkImage(light_image=self.neutral, size=(100, 100))
            self.result_text.configure(text="The model predicted the \nreview to be neutral.")
            self.result_text.grid_configure(row=0, column=0, padx=(92, 0), pady=(240, 0), sticky="nw")

        elif value == 2:
            self.result_image = ctk.CTkImage(light_image=self.happy, size=(100, 100))
            self.result_text.configure(text="The model predicted the \nreview to be positive.")
            self.result_text.grid_configure(row=0, column=0, padx=(92, 0), pady=(240, 0), sticky="nw")

        else:
            raise ValueError("Value must be 0, 1, or 2.")

        self.result_image_show.configure(image=self.result_image)


    def update_upvotes_label(self, value):
        self.upvotes_value_label.configure(text=int(float(value)))
    

    def update_total_votes_label(self, value):
        self.total_votes_value_label.configure(text=int(float(value)))


    def update_rating_label(self, value):
        self.rating_value_label.configure(text=int(float(value)))


    def create_data_visualization_tab(self):
        self.tab("Data Visualization").grid_columnconfigure((0, 1), weight=1)
        self.tab("Data Visualization").grid_rowconfigure((0, 1), weight=1)

        self.generation_frame = ctk.CTkFrame(master=self.tab("Data Visualization"), width=300, height=596, corner_radius=20, border_width=2, bg_color="transparent")
        self.generation_frame.grid(row=0, column=0, columnspan=2, padx=(60, 0), pady=(20, 0), sticky="nw")
        self.generation_frame.grid_propagate(False)
        self.generation_frame.grid_columnconfigure(0, weight=1)
        self.generation_frame.grid_rowconfigure((0, 1), weight=1)
        self.generation_frame_label = ctk.CTkLabel(master=self.generation_frame, text="Settings", font=("Arial", 24))
        self.generation_frame_label.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), sticky="nw")

        self.figure_options = ctk.CTkOptionMenu(master=self.generation_frame, width=200, corner_radius=5, values=["Feature Importance", "Polarity vs. Subjectivity", "Most Frequent Words", "Reaction on Keyword"])
        self.figure_options.grid(row=0, column=0, padx=(50, 0), pady=(160, 0), sticky="nw")
        self.figure_options_label = ctk.CTkLabel(master=self.generation_frame, text="Select figure type:", font=("Arial", 13), fg_color="transparent")
        self.figure_options_label.grid(row=0, column=0, padx=(50, 0), pady=(130, 0), sticky="nw")

        self.generate_button = ctk.CTkButton(master=self.generation_frame, width=150, height=40, corner_radius=15, text="Generate", font=("Arial", 18), anchor="center", command=self.generate_plot)
        self.generate_button.grid(row=1, column=0, padx=(76, 0), pady=(0, 50), sticky="sw")

        self.visualization_frame = ctk.CTkFrame(master=self.tab("Data Visualization"), width=920, height=596, corner_radius=20, border_width=2, bg_color="transparent")
        self.visualization_frame.grid(row=0, column=0, columnspan=2, padx=(0, 60), pady=(20, 0), sticky="ne")
        self.visualization_frame.grid_propagate(False)
        self.visualization_frame.grid_columnconfigure(0, weight=1)
        self.visualization_frame.grid_rowconfigure((0, 1), weight=1)
        self.visualization_frame_label = ctk.CTkLabel(master=self.visualization_frame, text="Visualization Window", font=("Arial", 24))
        self.visualization_frame_label.grid(row=0, column=0, padx=(0, 20), pady=(20, 0), sticky="ne")

        self.visualization_canvas_frame = ctk.CTkFrame(master=self.visualization_frame, width=800, height=500, bg_color="transparent", fg_color="transparent")
        self.visualization_canvas_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.visualization_frame.grid_propagate(False)
        self.visualization_frame.grid_columnconfigure(0, weight=1)
        self.visualization_frame.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots()
        plot_importance(current_system_model, ax=self.ax)
        plt.tight_layout() 
        self.visualization_canvas =  FigureCanvasTkAgg(self.fig, master=self.visualization_canvas_frame)
        self.visualization_canvas.draw()
        self.visualization_canvas_widget = self.visualization_canvas.get_tk_widget()
        self.visualization_canvas_widget.config(width=880, height=500)
        self.visualization_canvas_widget.grid(row=0, column=0, sticky="nsew")


    def generate_plot(self):
        current_option = self.figure_options.get()

        self.ax.clear()
        self.visualization_canvas.get_tk_widget().delete("all")

        if current_option == "Feature Importance":
            self.fig, self.ax = plt.subplots()
            plot_importance(current_system_model, ax=self.ax)

        elif current_option == "Polarity vs. Subjectivity":
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
            plot_polarity_subjectivity(df, ax1=self.ax1, ax2=self.ax2)

        elif current_option == "Most Frequent Words":
            self.fig, self.ax = plt.subplots()
            plot_most_frequent(df, ax=self.ax)

        elif current_option == "Reaction on Keyword":
            pass
        
        plt.tight_layout() 
        self.visualization_canvas =  FigureCanvasTkAgg(self.fig, master=self.visualization_canvas_frame)
        self.visualization_canvas.draw()
        self.visualization_canvas_widget = self.visualization_canvas.get_tk_widget()
        self.visualization_canvas_widget.config(width=880, height=500)
        self.visualization_canvas_widget.grid(row=0, column=0, sticky="nsew")


    def create_model_comparison_tab(self):
        self.tab("Model Comparison").grid_columnconfigure((0, 1), weight=1)
        self.tab("Model Comparison").grid_rowconfigure((0, 1), weight=1)

        self.configuration_frame = ctk.CTkFrame(master=self.tab("Model Comparison"), width=300, height=596, corner_radius=20, border_width=2, bg_color="transparent")
        self.configuration_frame.grid(row=0, column=0, columnspan=2, padx=(0, 60), pady=(20, 0), sticky="ne")
        self.configuration_frame.grid_propagate(False)
        self.configuration_frame.grid_columnconfigure(0, weight=1)
        self.configuration_frame.grid_rowconfigure((0, 1), weight=1)
        self.configuration_frame_label = ctk.CTkLabel(master=self.configuration_frame, text="Settings", font=("Arial", 24))
        self.configuration_frame_label.grid(row=0, column=0, padx=(0, 20), pady=(20, 0), sticky="ne")

        self.dataframe_options = ctk.CTkOptionMenu(master=self.configuration_frame, width=200, corner_radius=5, values=["Feature Importance", "Polarity vs. Subjectivity", "Most Frequent Words","Reaction on Keyword",])
        self.dataframe_options.grid(row=0, column=0, padx=(50, 0), pady=(160, 0), sticky="nw")
        self.dataframe_options_label = ctk.CTkLabel(master=self.configuration_frame, text="Select figure type:", font=("Arial", 13), fg_color="transparent")
        self.dataframe_options_label.grid(row=0, column=0, padx=(50, 0), pady=(130, 0), sticky="nw")

        self.generate_button = ctk.CTkButton(master=self.configuration_frame, width=150, height=40, corner_radius=15, text="Generate", font=("Arial", 18), anchor="center", command=self.generate_plot)
        self.generate_button.grid(row=1, column=0, padx=(76, 0), pady=(0, 50), sticky="sw")

        self.dataset_frame = ctk.CTkFrame(master=self.tab("Model Comparison"), width=920, height=596, corner_radius=20, border_width=2, bg_color="transparent")
        self.dataset_frame.grid(row=0, column=0, columnspan=2, padx=(60, 0), pady=(20, 0), sticky="nw")
        self.dataset_frame.grid_propagate(False)
        self.dataset_frame.grid_columnconfigure(0, weight=1)
        self.dataset_frame.grid_rowconfigure((0, 1), weight=1)
        self.dataset_frame_label = ctk.CTkLabel(master=self.dataset_frame, text="Dataframe Window", font=("Arial", 24))
        self.dataset_frame_label.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), sticky="nw")

        self.dataset_canvas_frame = ctk.CTkFrame(master=self.dataset_frame, width=800, height=500, bg_color="transparent", fg_color="transparent")
        self.dataset_canvas_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.dataset_frame.grid_propagate(False)
        self.dataset_frame.grid_columnconfigure(0, weight=1)
        self.dataset_frame.grid_rowconfigure(0, weight=1)


def plot_polarity_subjectivity(df, ax1, ax2):
    sns.histplot(df['polarity'], ax=ax1)
    sns.histplot(df['subjectivity'], ax=ax2)

    plt.suptitle('Distribution of Polarity and Subjectivity')

    
def plot_most_frequent(df, ax):
    df['review'] = df['review'].apply(lambda x: contractions.fix(x))
    cv = CountVectorizer(stop_words = 'english')
    words = cv.fit_transform(df['review'])
    sum_words = words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

    color = plt.cm.ocean(np.linspace(0, 1, 21))
    frequency.head(20).plot(x='word', y='freq', kind='bar', color = color, ax=ax)
    plt.title("Most Frequently Occuring Words - Top 20")

def plot_model_comparison():
    reviews = df['review'].copy() 
    sentiments = df['sentiment'].copy() 
    comparison_df = df.drop(['review', 'sentiment'], axis='columns')

    comparison_X = comparison_df
    comparison_y_pred = current_system_model.predict(comparison_X)
    
    comparison_df['review'] = reviews 
    comparison_df['sentiment'] = sentiments
    comparison_df['comparison_sentiment'] = comparison_y_pred


def process_user_review():
    user_review_value = app.tab_view.user_review_textbox.get("0.0","end")

    polarity = TextBlob(user_review_value).sentiment.polarity
    subjectivity = TextBlob(user_review_value).sentiment.subjectivity
    
    for result in classifier(user_review_value, truncation='longest_first', max_length=512):
        confidence = result['score']
        sentiment = result['label']
    
    return polarity, subjectivity, sentiment, confidence


def process_features():
    upvotes_value = app.tab_view.upvotes_slider.get()
    total_votes_value = app.tab_view.total_votes_slider.get()
    rating_value = app.tab_view.rating_slider.get()

    return upvotes_value, total_votes_value, rating_value


def predict():
    polarity, subjectivity, sentiment, confidence = process_user_review()
    upvotes_value, total_votes_value, rating_value = process_features()

    data = pd.DataFrame({
                        'upvotes': [upvotes_value], 
                        'total_votes': [total_votes_value], 
                        'rating': [rating_value], 
                        'polarity': [polarity],
                        'subjectivity': [subjectivity],
                        'sentiment': [sentiment], 
                        'confidence': [confidence]
                        })
    
    
    X = data.drop(columns=['sentiment'])
    y = data['sentiment']
    
    single_row = pd.DataFrame(X.iloc[0]).transpose()
    y_pred = current_system_model.predict(single_row)

    app.tab_view.update_sentiment_display(y_pred)


if __name__ == "__main__":
    df = pd.read_csv('dataset/reviews_preprocessed.csv')

    #'''
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    #'''
    current_system_model = joblib.load('model/main/xgbert.pkl')

    print("\n\nSystem loaded successfully...")

    app = App()
    app.mainloop()

    print("\n\nSystem exited...\n\n")

