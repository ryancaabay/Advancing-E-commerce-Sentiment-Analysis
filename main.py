print("\n\nLoading system...\n\n")


import joblib
import numpy as np
import contractions
import pandas as pd
import seaborn as sns
from PIL import Image
from tkinter import ttk
import customtkinter as ctk
from textblob import TextBlob
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
        self.upvotes_slider = ctk.CTkSlider(master=self.sliders_frame, height=220, from_=0, to=10, number_of_steps=10, orientation="vertical", command=lambda value: self.update_feature_label(self.upvotes_value_label, value))
        self.upvotes_slider.grid(row=1, column=0, padx=(60, 0), pady=(0, 90), sticky="sw")
        self.upvotes_value_label = ctk.CTkLabel(master=self.sliders_frame, text="5", font=("Arial", 13))
        self.upvotes_value_label.grid(row=1, column=0, padx=(2, 0), pady=(0, 60), sticky="swe")

        self.total_votes_label = ctk.CTkLabel(master=self.sliders_frame, text="Total Votes", font=("Arial", 13))
        self.total_votes_label.grid(row=0, column=1, padx=(39, 0), pady=(120, 0), sticky="sw")
        self.total_votes_slider = ctk.CTkSlider(master=self.sliders_frame, height=220, from_=0, to=10, number_of_steps=10, orientation="vertical", command=lambda value: self.update_feature_label(self.total_votes_value_label, value))
        self.total_votes_slider.grid(row=1, column=1, padx=(64, 0), pady=(0, 90), sticky="sw")
        self.total_votes_value_label = ctk.CTkLabel(master=self.sliders_frame, text="5", font=("Arial", 13))
        self.total_votes_value_label.grid(row=1, column=1, padx=(0, 4), pady=(0, 60), sticky="swe")
        
        self.rating_label = ctk.CTkLabel(master=self.sliders_frame, text="Rating", font=("Arial", 13))
        self.rating_label.grid(row=0, column=2, padx=(38, 0), pady=(120, 0), sticky="sw")
        self.rating_slider = ctk.CTkSlider(master=self.sliders_frame, height=220, from_=1, to=5, number_of_steps=4, orientation="vertical", command=lambda value: self.update_feature_label(self.rating_value_label, value))
        self.rating_slider.grid(row=1, column=2, padx=(48, 0), pady=(0, 90), sticky="sw")
        self.rating_value_label = ctk.CTkLabel(master=self.sliders_frame, text="3", font=("Arial", 13))
        self.rating_value_label.grid(row=1, column=2, padx=(0, 6), pady=(0, 60), sticky="swe")

        self.user_review_textbox = ctk.CTkTextbox(master=self.tab("Sentiment Analysis"), width=1200, height=100, corner_radius=20, border_width=2, fg_color="transparent", font=("Arial", 18), wrap="word")
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
        self.accuracy_score_bar.set(value=0.00)
        self.accuracy_score_percentage = ctk.CTkLabel(master=self.results_frame, text="0%", font=("Arial", 13), bg_color="transparent")
        self.accuracy_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(139, 0), sticky="ne")
        self.after(1000, self.animate_metric_bar, self.accuracy_score_bar, accuracy_score)
        self.after(1000, self.animate_metric_label, self.accuracy_score_percentage, accuracy_score)

        self.precision_score_label = ctk.CTkLabel(master=self.results_frame, text="Precision", font=("Arial", 13), bg_color="transparent")
        self.precision_score_label.grid(row=0, column=1, padx=(0, 376), pady=(180, 0), sticky="ne")
        self.precision_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.precision_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(210, 0), sticky="ne")
        self.precision_score_bar.set(value=0.00)
        self.precision_score_percentage = ctk.CTkLabel(master=self.results_frame, text="0%", font=("Arial", 13), bg_color="transparent")
        self.precision_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(199, 0), sticky="ne")
        self.after(1000, self.animate_metric_bar, self.precision_score_bar, precision_score)
        self.after(1000, self.animate_metric_label, self.precision_score_percentage, precision_score)

        self.recall_score_label = ctk.CTkLabel(master=self.results_frame, text="Recall", font=("Arial", 13), bg_color="transparent")
        self.recall_score_label.grid(row=0, column=1, padx=(0, 394), pady=(240, 0), sticky="ne")
        self.recall_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.recall_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(270, 0), sticky="ne")
        self.recall_score_bar.set(value=0.00)
        self.recall_score_percentage = ctk.CTkLabel(master=self.results_frame, text="0%", font=("Arial", 13), bg_color="transparent")
        self.recall_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(259, 0), sticky="ne")
        self.after(1000, self.animate_metric_bar, self.recall_score_bar, recall_score)
        self.after(1000, self.animate_metric_label, self.recall_score_percentage, recall_score)

        self.f1_score_label = ctk.CTkLabel(master=self.results_frame, text="F1", font=("Arial", 13), bg_color="transparent")
        self.f1_score_label.grid(row=0, column=1, padx=(0, 414), pady=(300, 0), sticky="ne")
        self.f1_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.f1_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(330, 0), sticky="ne")
        self.f1_score_bar.set(value=0.00)
        self.f1_score_percentage = ctk.CTkLabel(master=self.results_frame, text="0%", font=("Arial", 13), bg_color="transparent")
        self.f1_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(319, 0), sticky="ne")
        self.after(1000, self.animate_metric_bar, self.f1_score_bar, f1_score)
        self.after(1000, self.animate_metric_label, self.f1_score_percentage, f1_score)

        self.cv_score_label = ctk.CTkLabel(master=self.results_frame, text="Cross-Validation", font=("Arial", 13), bg_color="transparent")
        self.cv_score_label.grid(row=0, column=1, padx=(0, 335), pady=(360, 0), sticky="ne")
        self.cv_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.cv_score_bar.grid(row=0, column=1, padx=(0, 80), pady=(390, 0), sticky="ne")
        self.cv_score_bar.set(value=0.00)
        self.cv_score_percentage = ctk.CTkLabel(master=self.results_frame, text="0%", font=("Arial", 13), bg_color="transparent")
        self.cv_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(379, 0), sticky="ne")
        self.after(1000, self.animate_metric_bar, self.cv_score_bar, cross_val_score)
        self.after(1000, self.animate_metric_label, self.cv_score_percentage, cross_val_score)

    
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


    def update_feature_label(self, label, value):
        label.configure(text=int(float(value)))
    

    def animate_metric_bar(self, metric_bar, metric_score):
        current_value = metric_bar.get()

        if current_value < metric_score:
            current_value += 0.01  
            metric_bar.set(current_value)
            app.after(20, self.animate_metric_bar, metric_bar, metric_score)  


    def animate_metric_label(self, metric_label, metric_score):
        current_value = int(metric_label.cget("text").replace('%', ''))

        if current_value < round(float(metric_score) * 100, 2):
            current_value += 1
            metric_label.configure(text=str(current_value) + "%")
            app.after(15, self.animate_metric_label, metric_label, metric_score)


    def create_data_visualization_tab(self):
        self.tab("Data Visualization").grid_columnconfigure((0, 1), weight=1)
        self.tab("Data Visualization").grid_rowconfigure((0, 1), weight=1)

        self.generation_frame = ctk.CTkFrame(master=self.tab("Data Visualization"), width=300, height=596, corner_radius=20, border_width=2, bg_color="transparent")
        self.generation_frame.grid(row=0, column=0, columnspan=2, padx=(60, 0), pady=(20, 0), sticky="nw")
        self.generation_frame.grid_propagate(False)
        self.generation_frame.grid_columnconfigure(0, weight=1)
        self.generation_frame.grid_rowconfigure((0, 1), weight=1)
        self.generation_frame_label = ctk.CTkLabel(master=self.generation_frame, text="Plot Type", font=("Arial", 24))
        self.generation_frame_label.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), sticky="nw")

        self.plot_options = ctk.CTkOptionMenu(master=self.generation_frame, width=200, corner_radius=5, values=["Feature Importance", "Polarity vs. Subjectivity", "Most Frequent Words", "Reaction on Keyword"])
        self.plot_options.grid(row=0, column=0, padx=(50, 0), pady=(160, 0), sticky="nw")
        self.plot_options_label = ctk.CTkLabel(master=self.generation_frame, text="Select visualization option:", font=("Arial", 13), fg_color="transparent")
        self.plot_options_label.grid(row=0, column=0, padx=(50, 0), pady=(130, 0), sticky="nw")

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
        self.visualization_canvas_frame.grid_propagate(False)
        self.visualization_canvas_frame.grid_columnconfigure(0, weight=1)
        self.visualization_canvas_frame.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots()
        plot_importance(current_model, ax=self.ax)

        self.visualization_canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_canvas_frame)
        self.visualization_canvas.draw()
        self.visualization_canvas_widget = self.visualization_canvas.get_tk_widget()
        self.visualization_canvas_widget.config(width=880, height=500)
        self.visualization_canvas_widget.grid(row=0, column=0, sticky="nsew")


    def generate_plot(self):
        current_option = self.plot_options.get()

        self.ax.clear()
        self.fig.clf()

        if current_option == "Feature Importance":
            self.fig, self.ax = plt.subplots()
            plot_importance(current_model, ax=self.ax)

        elif current_option == "Polarity vs. Subjectivity":
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
            plot_polarity_subjectivity(df, ax1=self.ax1, ax2=self.ax2)

        elif current_option == "Most Frequent Words":
            self.fig, self.ax = plt.subplots()
            plot_most_frequent(df, ax=self.ax)
            plt.tight_layout() 

        elif current_option == "Reaction on Keyword":
            self.keyword_input = ctk.CTkInputDialog(text="Type in desired keyword/s:", title="Reaction on Keyword")
            self.keyword = self.keyword_input.get_input()
            self.fig, self.ax = plt.subplots()
            plot_reaction_on_keyword(df, self.keyword, ax=self.ax)
            plt.tight_layout()
        
        self.visualization_canvas =  FigureCanvasTkAgg(self.fig, master=self.visualization_canvas_frame)
        self.visualization_canvas.draw()
        self.visualization_canvas_widget = self.visualization_canvas.get_tk_widget()
        self.visualization_canvas_widget.config(width=880, height=500)
        self.visualization_canvas_widget.grid(row=0, column=0, sticky="nsew")


    def create_model_comparison_tab(self):
        self.tab("Model Comparison").grid_columnconfigure((0, 1), weight=1)
        self.tab("Model Comparison").grid_rowconfigure((0, 1), weight=1)

        self.search_filter_frame = ctk.CTkFrame(master=self.tab("Model Comparison"), width=300, height=596, corner_radius=20, border_width=2, bg_color="transparent")
        self.search_filter_frame.grid(row=0, column=0, columnspan=2, padx=(0, 60), pady=(20, 0), sticky="ne")
        self.search_filter_frame.grid_propagate(False)
        self.search_filter_frame.grid_columnconfigure(0, weight=1)
        self.search_filter_frame.grid_rowconfigure((0, 1), weight=1)
        self.search_filter_frame_label = ctk.CTkLabel(master=self.search_filter_frame, text="Search Filter", font=("Arial", 24))
        self.search_filter_frame_label.grid(row=0, column=0, padx=(0, 20), pady=(20, 0), sticky="ne")

        self.search_entry = ctk.CTkEntry(master=self.search_filter_frame, width=200, corner_radius=5, font=("Arial", 13))
        self.search_entry.grid(row=0, column=0, padx=(50, 0), pady=(160, 0), sticky="nw")
        #self.search_entry.bind('<KeyRelease>', self.filter_search) #
        self.search_entry_label = ctk.CTkLabel(master=self.search_filter_frame, text="Filter data by keyword:", font=("Arial", 13), fg_color="transparent")
        self.search_entry_label.grid(row=0, column=0, padx=(50, 0), pady=(130, 0), sticky="nw")

        self.review_text = ctk.CTkTextbox(master=self.search_filter_frame, width=200, height=300, corner_radius=5, border_width=2, fg_color="transparent", font=("Arial", 18), wrap="word")
        self.review_text.grid(row=0, column=0, padx=(50, 0), pady=(220, 0), sticky="nw")
        self.review_text.insert("1.0", "Review text of selected row goes here...")
        self.review_text.configure(state="disabled")

        self.dataset_frame = ctk.CTkFrame(master=self.tab("Model Comparison"), width=920, height=596, corner_radius=20, border_width=2, bg_color="transparent")
        self.dataset_frame.grid(row=0, column=0, columnspan=2, padx=(60, 0), pady=(20, 0), sticky="nw")
        self.dataset_frame.grid_propagate(False)
        self.dataset_frame.grid_columnconfigure(0, weight=1)
        self.dataset_frame.grid_rowconfigure((0, 1), weight=1)
        self.dataset_frame_label = ctk.CTkLabel(master=self.dataset_frame, text="Dataframe Window", font=("Arial", 24))
        self.dataset_frame_label.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), sticky="nw")

        self.dataset_canvas_frame = ctk.CTkFrame(master=self.dataset_frame, width=800, height=500, bg_color="transparent", fg_color="transparent")
        self.dataset_canvas_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.dataset_canvas_frame.grid_propagate(False)
        self.dataset_canvas_frame.grid_columnconfigure(0, weight=1)
        self.dataset_canvas_frame.grid_rowconfigure(0, weight=1)

        generate_model_comparison()
    '''
        self.comparison_dataframe = pd.read_csv('dataset/model_comparison.csv')
        self.column_names = list(self.comparison_dataframe)  
        self.result_tree_view = ttk.Treeview(master=self.dataset_canvas_frame, selectmode='browse')
        self.result_tree_view.grid(row=0, column=0, sticky="nsew")
        self.result_tree_view['show'] = 'headings'
        self.result_tree_view['columns'] = self.column_names

        for self.column_name in self.column_names:
            self.result_tree_view.column(self.column_name, width=30, anchor='c')
            self.result_tree_view.heading(self.column_name, text=self.column_name)

        self.data_list = self.comparison_dataframe.values.tolist()

        for index, row in self.comparison_dataframe.iterrows():
            index = str(index)
            row.iloc[-1] = str(row.iloc[-1])
            values = list(row.values)
            self.result_tree_view.insert("", 'end', iid=index, values=values)
        
        self.result_tree_view.bind('<<TreeviewSelect>>', self.display_review_text)
        

    def filter_search(self, event):
        self.result_tree_view.selection_remove(self.result_tree_view.selection())

        for all_rows in self.result_tree_view.get_children():
            self.result_tree_view.delete(all_rows)

        search_parts = self.search_entry.get().strip().lower().split(' ')

        if len(search_parts) % 2 == 0:
            # if search query contains an even number of parts, treat them as column-value pairs
            mask = pd.Series([True]*len(self.comparison_dataframe))
            for i in range(0, len(search_parts), 2):
                column, search_word = search_parts[i], search_parts[i+1]
                mask = mask & self.comparison_dataframe[column].astype(str).str.lower().str.contains(search_word)
            self.filtered_dataframe = self.comparison_dataframe[mask]
        else:
            # if search query does not contain an even number of parts, treat it as a general search
            if all(word.isdigit() for word in search_parts):
                mask = self.comparison_dataframe.astype(str).apply(lambda row: all(word in row.values for word in search_parts) or any(word in str(row.name).lower() for word in search_parts), axis=1)
            else:
                mask = self.comparison_dataframe.apply(lambda row: all(any(word in str(value).lower() for value in row.values) for word in search_parts) or any(word in str(row.name).lower() for word in search_parts), axis=1)
            self.filtered_dataframe = self.comparison_dataframe[mask]

        for index, row in self.filtered_dataframe.iterrows():
            values = list(row.values)
            self.result_tree_view.insert("", 'end', values=values)
    '''

    def display_review_text(self, event):
        selected_items = self.result_tree_view.selection()

        if selected_items:  
            selected_item = selected_items[0]
            item_values = self.result_tree_view.item(selected_item, 'values')

            if len(item_values) > 7:  
                review_text_value = item_values[7]

                self.review_text.configure(state="normal")
                self.review_text.delete("1.0", "end")

                self.review_text.insert("1.0", review_text_value)
                self.review_text.configure(state="disabled")


def plot_polarity_subjectivity(df, ax1, ax2):
    sns.histplot(df['polarity'], ax=ax1)
    sns.histplot(df['subjectivity'], ax=ax2)

    plt.suptitle('Distribution of Polarity and Subjectivity')

    
def plot_most_frequent(df, ax):
    df['review'] = df['review'].apply(lambda x: contractions.fix(x))
    count_vector = CountVectorizer(stop_words = 'english')
    words = count_vector.fit_transform(df['review'])
    sum_words = words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in count_vector.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

    color = plt.cm.ocean(np.linspace(0, 1, 21))
    frequency.head(20).plot(x='word', y='freq', kind='bar', color = color, ax=ax)
    plt.title("Most Frequently Occuring Words - Top 20")

                
def plot_reaction_on_keyword(df, keyword, ax):
    df = mc.copy()


    def get_xgbert_value(value):
        if value == 0:
            return "Negative"
        elif value == 1:
            return "Neutral"
        elif value == 2:
            return "Positive"
        

    def percentage(part, whole):
        if whole == 0:
            return 0.0
        return 100 * float(part) / float(whole)
    

    def custom_autopct(pct):
        return ('%1.1f%%' % pct) if pct > 0 else ''


    df['xgbert'] = df['xgbert'].apply(get_xgbert_value)

    searched_term = keyword.lower()

    df = df[df['review'].str.lower().str.contains(searched_term, na=False)]

    negative = len(df[df['xgbert'] == 'Negative'])
    neutral = len(df[df['xgbert'] == 'Neutral'])
    positive = len(df[df['xgbert'] == 'Positive'])

    total = negative + neutral + positive

    if total == 0:
        print("No reviews found for the keyword.")
        return

    positive = percentage(positive, total)
    negative = percentage(negative, total)
    neutral = percentage(neutral, total)

    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'gold', 'red']
    labels = ['Positive', 'Neutral', 'Negative']
    
    ax.pie(sizes, colors = colors, autopct=custom_autopct)
    ax.legend(labels, loc='lower left')
    ax.set_title('Reaction of People on the Keyword \'' + searched_term + '\'' + ' which appeared ' + str(total) + ' time/s.')
    ax.axis('equal')


def generate_model_comparison():
    reviews = df['review'].copy() 
    sentiments = df['sentiment'].copy() 
    comparison_df = df.drop(['review', 'sentiment'], axis='columns')

    comparison_X = comparison_df
    comparison_y_pred = current_model.predict(comparison_X)
    
    comparison_df['review'] = reviews 
    comparison_df['bert'] = sentiments
    comparison_df['xgbert'] = comparison_y_pred

    comparison_df.reset_index(inplace=True)
    comparison_df.rename(columns={'index': 'id'}, inplace=True)

    if 'polarity' in comparison_df.columns:
        comparison_df['polarity'] = comparison_df['polarity'].round(2)
    if 'subjectivity' in comparison_df.columns:
        comparison_df['subjectivity'] = comparison_df['subjectivity'].round(2)
    if 'confidence' in comparison_df.columns:
        comparison_df['confidence'] = comparison_df['confidence'].round(2)

    comparison_df.to_csv('dataset/model_comparison.csv', index=False)


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
    y_pred = current_model.predict(single_row)

    app.tab_view.update_sentiment_display(y_pred)


if __name__ == "__main__":
    df = pd.read_csv('dataset/reviews_preprocessed.csv')
    #mc = pd.read_csv('dataset/model_comparison.csv') #

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    current_model, X, y, X_test, y_test = (joblib.load(f'model/main/{name}') for name in 
                                                             ['xgbert.pkl', 'X.pkl', 'y.pkl', 'X_test.pkl', 'y_test.pkl'])
    
    y_pred = current_model.predict(X_test)

    accuracy_score = accuracy_score(y_test, y_pred,)
    precision_score = precision_score(y_test,y_pred, average='weighted')
    recall_score = recall_score(y_test,y_pred, average='weighted')
    f1_score = f1_score(y_test, y_pred, average='weighted')
    cross_val_score = cross_val_score(current_model, X, y, cv=10).mean()

    app = App()

    print("\n\nSystem loaded successfully...")

    app.mainloop()

    print("\n\nSystem exited...\n\n")