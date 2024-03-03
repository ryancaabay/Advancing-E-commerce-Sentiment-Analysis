import customtkinter as ctk
from PIL import Image

class MyTabView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.add("Sentiment Analyzer")  
        self.add("Data Visualization")
        self.add("Model Comparison")  

        self.create_sentiment_analyzer_tab()
        self.create_data_visualization_tab()
        self.create_model_comparison_tab()

    def create_sentiment_analyzer_tab(self):
        self.tab("Sentiment Analyzer").grid_columnconfigure((0, 1), weight=1)
        self.tab("Sentiment Analyzer").grid_rowconfigure((0, 1), weight=1)
        
        self.sliders_frame = ctk.CTkFrame(master=self.tab("Sentiment Analyzer"), width=400, height=450, corner_radius=20, border_width=2, bg_color="transparent")
        self.sliders_frame.grid(row=0, column=0, columnspan=3, padx=(60, 0), pady=(20, 0), sticky="nw")
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

        self.textbox = ctk.CTkTextbox(master=self.tab("Sentiment Analyzer"), width=1200, height=100, corner_radius=20, border_width=2, fg_color="transparent", font=("Arial", 18))
        self.textbox.grid(row=1, column=0, padx=(60, 0), pady=(20, 40), sticky="sw")
        self.textbox.insert("0.0", "Enter your review here...")
        self.textbox.bind("<FocusIn>", lambda event: self.textbox.delete("0.0", "end"))

        self.predict_button = ctk.CTkButton(master=self.tab("Sentiment Analyzer"), width=50, height=50, corner_radius=15, border_width=0, text="↑", font=("Arial", 24), anchor="center")
        self.predict_button.grid(row=1, column=1, padx=(0, 60), pady=(0, 65), sticky="se")

        self.results_frame = ctk.CTkFrame(master=self.tab("Sentiment Analyzer"), width=820, height=450, corner_radius=20, border_width=2, bg_color="transparent")
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
        self.result_text = ctk.CTkLabel(master=self.results_frame, text="Hi there, input some review\n and I will try to predict it!", font=("Arial", 18), bg_color="transparent")
        self.result_text.grid(row=0, column=0, padx=(80, 0), pady=(240, 0), sticky="nw")

        self.accuracy_score_label = ctk.CTkLabel(master=self.results_frame, text="Accuracy", font=("Arial", 13), bg_color="transparent")
        self.accuracy_score_label.grid(row=0, column=1, padx=(25, 0), pady=(120, 0), sticky="nw")
        self.accuracy_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.accuracy_score_bar.grid(row=0, column=1, padx=(25, 0), pady=(150, 0), sticky="nw")
        self.accuracy_score_bar.set(0.88)
        self.accuracy_score_percentage = ctk.CTkLabel(master=self.results_frame, text="88%", font=("Arial", 13), bg_color="transparent")
        self.accuracy_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(139, 0), sticky="ne")

        self.precision_score_label = ctk.CTkLabel(master=self.results_frame, text="Precision", font=("Arial", 13), bg_color="transparent")
        self.precision_score_label.grid(row=0, column=1, padx=(25, 0), pady=(180, 0), sticky="nw")
        self.precision_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.precision_score_bar.grid(row=0, column=1, padx=(25, 0), pady=(210, 0), sticky="nw")
        self.precision_score_bar.set(0.86)
        self.precision_score_percentage = ctk.CTkLabel(master=self.results_frame, text="86%", font=("Arial", 13), bg_color="transparent")
        self.precision_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(199, 0), sticky="ne")

        self.recall_score_label = ctk.CTkLabel(master=self.results_frame, text="Recall", font=("Arial", 13), bg_color="transparent")
        self.recall_score_label.grid(row=0, column=1, padx=(25, 0), pady=(240, 0), sticky="nw")
        self.recall_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.recall_score_bar.grid(row=0, column=1, padx=(25, 0), pady=(270, 0), sticky="nw")
        self.recall_score_bar.set(0.84)
        self.recall_score_percentage = ctk.CTkLabel(master=self.results_frame, text="84%", font=("Arial", 13), bg_color="transparent")
        self.recall_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(259, 0), sticky="ne")

        self.f1_score_label = ctk.CTkLabel(master=self.results_frame, text="F1", font=("Arial", 13), bg_color="transparent")
        self.f1_score_label.grid(row=0, column=1, padx=(25, 0), pady=(300, 0), sticky="nw")
        self.f1_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.f1_score_bar.grid(row=0, column=1, padx=(25, 0), pady=(330, 0), sticky="nw")
        self.f1_score_bar.set(0.82)
        self.f1_score_percentage = ctk.CTkLabel(master=self.results_frame, text="82%", font=("Arial", 13), bg_color="transparent")
        self.f1_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(319, 0), sticky="ne")

        self.cv_score_label = ctk.CTkLabel(master=self.results_frame, text="Cross-Validation", font=("Arial", 13), bg_color="transparent")
        self.cv_score_label.grid(row=0, column=1, padx=(25, 0), pady=(360, 0), sticky="nw")
        self.cv_score_bar = ctk.CTkProgressBar(master=self.results_frame, width=350, orientation="horizontal")
        self.cv_score_bar.grid(row=0, column=1, padx=(25, 0), pady=(390, 0), sticky="nw")
        self.cv_score_bar.set(0.86)
        self.cv_score_percentage = ctk.CTkLabel(master=self.results_frame, text="86%", font=("Arial", 13), bg_color="transparent")
        self.cv_score_percentage.grid(row=0, column=1, padx=(0, 40), pady=(379, 0), sticky="ne")


        #self.update_sentiment_display(1)
    
    def update_sentiment_display(self, value):
        self.happy = Image.open("images/happy.png")
        self.neutral = Image.open("images/neutral.png")
        self.sad = Image.open("images/sad.png")

        if value == 0:
            self.result_image = ctk.CTkImage(light_image=self.sad, size=(100, 100))
            self.result_text.configure(text="The model predicted the \nreview to be negative.")
            self.result_text.grid_configure(row=0, column=0, padx=(90, 0), pady=(240, 0), sticky="nw")
        elif value == 1:
            self.result_image = ctk.CTkImage(light_image=self.neutral, size=(100, 100))
            self.result_text.configure(text="The model predicted the \nreview to be neutral.")
            self.result_text.grid_configure(row=0, column=0, padx=(90, 0), pady=(240, 0), sticky="nw")
        elif value == 2:
            self.result_image = ctk.CTkImage(light_image=self.happy, size=(100, 100))
            self.result_text.configure(text="The model predicted the \nreview to be positive.")
            self.result_text.grid_configure(row=0, column=0, padx=(90, 0), pady=(240, 0), sticky="nw")
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
        self.label = ctk.CTkLabel(master=self.tab("Data Visualization"))
        self.label.grid(row=0, column=0, padx=20, pady=10)

    def create_model_comparison_tab(self):
        self.label = ctk.CTkLabel(master=self.tab("Model Comparison"))
        self.label.grid(row=0, column=0, padx=20, pady=10)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sentiment Analysis Tool")
        self.geometry("1440x810")
        self.resizable(False, False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        
        self.tab_view = MyTabView(master=self, width=1400, height=700)
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


app = App()
app.mainloop()

