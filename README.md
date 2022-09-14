MAESTRO-Deep-Learning-Approaches-to-Melody-Reduction
=========================================
This is the final project of GCT634 : Musical Applications of Machine Learning.

(https://drive.google.com/file/d/1fi1ydA8GyhhVYWBioYLtTOX08QnjIS6c/view?usp=sharing)

Collaborator: Cheol-gyu Jin

---
ABSTRACT

We are exposed to a great deal of music. As a result, many people have a desire to learn to play music. However, it takes a long time to master the instrument. Many people give up in this process. We want to provide a sense of accomplishment by providing an easy musical 
score of the song they want. If so, more people will be able to continue learning. We used the GTTM dataset(https://gttm.jp/gttm/database/) to create an easy score ground truth dataset. We propose a LSTM model that provides easy score.

---

Melody Reduction

The time-span reduction represents an intuitive idea: if we remove ornamental notes from a long melody, we obtain a simpler, similar sounding melody. An entire piece 
of tonal music can eventually be reduced to one important note.

---

Data Preparation

The attributes element contains key information needed to interpret the notes and musical data that follow in this part. Each note in MusicXML has a duration element. The divisions element provided the unit of measure for the duration element in terms of divisions per quarter note. Since all we have in this file is one whole note, we never have to divide a quarter note, so we set the divisions value to 1.

Musical durations are typically expressed as fractions, such as "quarter" and "eighth" notes. MusicXML durations are fractions, too. Since the denominator rarely needs to change, it is represented separately in the divisions element, so that only the numerator needs to be associated with each individual note. This is similar to the scheme used in MIDI to represent note durations.

The key element is used to represent a key signature. Here we are in the key of C major, with no flats or sharps, so the fifths element is 0. If we were in the key of D major with 2 sharps, fifths would be set to 2. If we were in the key of F major with 1 flat, fifths would be set to -1. The name "fifths" comes from the representation of a key signature along the circle of fifths. It lets us represent standard key signatures with one element, instead of separate elements for sharps and flats.

The time element represents a time signature. Its two component elements, beat and beat-type, are the numerator and denominator of the time signature, respectively.

The pitch element must have a step and an octave element. Optionally it can have an alter element, if there is a flat or sharp involved. These elements represent sound, so the alter element must always be included if used, even if the alteration is in the key signature. In this case, we have no alteration. The pitch step is C. The octave of 4 indicatesthe octave the starts with middle C. Thus this note is a middle C. After understanind the structure of musicXML,we implemented parser using python. We decided to use 9 features for each note..

---

Our LSTM approaches

We use Musical XML file as the input of LSTM. The LSTM models classified each node to delete or not. For example, when classifier select 0 for some node it means that selected node will deleted in reduced melody. The LSTM model should classify each node, we select many-to-many LSTM classification model. We use results of rule-based melody reduction algorithms as ground-truth.

---

Evaluation

We plot the loss and accuracy with tensorboard. Xdimension is the number of epoch and y-dimension is target value. Our training set is 80 songs in GTTM datasets,
and test set is 20 songs. Accuracy is computed to compare each nodes. We select the model that has best results on test set to avoid overfitting.

![image](https://user-images.githubusercontent.com/64250761/190186195-06ee99c1-02be-4694-8664-4fa945844534.png)

---

Results

![image](https://user-images.githubusercontent.com/64250761/190185764-157d8e2f-aa5e-4f9f-8cb2-2a4ee6e12fcd.png)

---

Future works

There are some results that show over-reduced. Overreduced means that the reduction offers a lot so the reduced song lost their identity. To deal with problem we 
can make “level” for reduction. The model suggests various reduced song that have different reduction level. Also, we can provide the level-by-level music sheet, that will very helpful to beginner.

---

References

[1] Ryan Groves : AUTOMATIC MELODIC REDUCTION USING A SUPERVISED PROBABILISTIC CONTEXT-FREE GRAMMAR

[2] Masatoshi Hamanaka : Polyphonic Music Analysis Database Based on GTTM.

[3] Alex Sherstinsky : Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network.

[4] Hamanaka, Masatoshi, and Satoshi Tojo : "Interactive GTTM Analyzer." ISMIR. 2009.

---
