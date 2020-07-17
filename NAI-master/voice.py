import speech_recognition as sr
import pyttsx3



def SpeakText(command):
	engine = pyttsx3.init()
	engine.say(command)
	engine.runAndWait()

def listen():
	r = sr.Recognizer()
	while(1):
		try:
			with sr.Microphone() as source2:
				print("started")

				r.adjust_for_ambient_noise(source2,duration=0.5)
				audio2 = r.listen(source2)
				MyText = r.recognize_google(audio2)
				MyText = MyText.lower()

				print("Did u say "+MyText)
				SpeakText("Did u say"+MyText)
		except sr.RequestError as e:
			print(" Could not request results")
		except sr.UnknownValueError:
			continue    	

		if "quit" in MyText:
			break
	print("Exited successfully")
listen()