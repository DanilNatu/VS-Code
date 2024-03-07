# mengimport library dataset yg dibutuhkan
import numpy as np
import cv2
import random

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# dataset yg dipakai
face_cascade = cv2.CascadeClassifier('dataxml\\haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('dataxml\\haarcascade_mcs_mouth.xml')




# nilai ambang atas untuk mengatur cahaya di kamera, jika dibawah 80 = hitam & sedangkan diatas 80 = putih
bw_threshold = 80

# jenis font dari message
font = cv2.FONT_HERSHEY_SIMPLEX

# titik koordinat
org = (30, 30)

# message yg ditampilkan jika menggunakan masker
weared_mask_font_color = (0, 255, 0)

# message yg ditampilkan jika wajah tidak terdeteksi
no_face_color = (255, 0, 0)

# message yg ditampilkan jika TIDAK menggunakan masker
not_weared_mask_font_color = (0, 0, 255)

#  ketebalan garis kotak (rectangle)
thickness = 2

# menentukan skala ukuran font atau teks yang akan digunakan saat menampilkan pesan pada gambar hasil deteksi masker
font_scale = 1

# hasil message ketika menggunakan masker
weared_mask = "Makasih y"

# hasil message ketika TIDAK menggunakan masker
not_weared_mask = "Pake masker ga!"

# cap merupakan variabel untuk membuka koneksi ke kamera utama dan membuat objek 
cap = cv2.VideoCapture(0)

# loop utama yang terus berjalan selama kondisi yang diberikan benar. Kondisi di sini adalah 1, yang selalu benar, sehingga loop akan berjalan tanpa henti.
while 1:
    ret, img = cap.read() # Baris ini membaca satu frame dari kamera dan menyimpannya dalam variabel img.
    # fungsi cap.read() mengembalikan dua nilai: ret (boolean yang menunjukkan apakah bacaan berhasil) dan img (frame yang dibaca).

    img = cv2.flip(img,1) # Ini memutar atau memflip frame secara horizontal menggunakan cv2.flip(). Hal ini dapat diperlukan tergantung pada orientasi kamera atau kebutuhan deteksi tertentu.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Baris ini mengonversi frame ke skala keabuan (grayscale). Deteksi wajah sering dilakukan lebih baik pada citra grayscale.

    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY) #  baris kode tersebut mengonversi gambar hitam putih (gray) menjadi gambar biner (hitam dan putih) dengan memutuskan warna setiap piksel berdasarkan nilai ambang batas (bw_threshold).

    # untuk mendeteksi cahaya
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # untuk mendeteksi objek
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)


    if(len(faces) == 0 and len(faces_bw) == 0):# jika variable faces dan faces_bw bernilai 0 maka kamera(cv2.putText) akan menampilkan frame wajah dengan teks "mana mukamu nich", titik koordinat, font yg sudah diatur, skala ukuran font, warna font, ketebalah, dan di tampilkan dengan kode cv2.LINE_AA
        cv2.putText(img, "mana mukamu nich", org, font, font_scale, no_face_color, thickness, cv2.LINE_AA)
    else:

        for (x, y, w, h) in faces:# print frame
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)# ngeprint bentuk dan ukuran frame, warna frame dan ketebalan frame
            roi_gray = gray[y:y + h, x:x + w]#menentukan intensitas cahaya pada frame
            roi_color = img[y:y + h, x:x + w]#menentukan warna objekpada frame


            #untuk mendeteksi mulut
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        # mendeteksi wajah tetapi tidak mendeteksi mulut sehingga orang tersebut memakai masker
        if(len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if(y < my < y + h):
                    # mendeteksi wajah dan mulut sehingga orang tersebut tidak memakai masker
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)

                    #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                    break

    # menampilkan hasil frame
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff #memberikan perintah untuk menunggu selama 30 milidetik
    if k == 27:# menutup frame jika tombol esc ditekan
        break

# Release video
cap.release()
cv2.destroyAllWindows()