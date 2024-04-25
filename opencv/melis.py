command = ""
started = False

while True:
    command = input(">").lower()
    if command == "start":
        if started:
            print("araba şu an da çalışıyor")
        else:
            started = True
            print("Araba çalıştırıldı balım")

    elif command == "stop":

        if not started:  #"şuan araba çalışıyor"
            print("şuan ki güncel started= ", started)
            print("Araba şu anda çalışmıyor")
        else:
            started = False
            print("Araba durduruldu")

    else:
        print("doğru şeyi yaz")