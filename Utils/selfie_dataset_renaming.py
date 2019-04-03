my_list = []
fail_count = 0
ok_count = 0

with open('selfie_dataset.txt', 'r') as f:
    data = f.read()
    files = data.split('\n')[:-1]
    
    for file in files:
        values = file.split(' ')
        values = values[0:1] + values[3:10]
        if(values[1] == '-1' or values[1] == '1'):
            newName = ('0' if values[1] == '-1' else '1') + '_'
            try:
                newName = newName + str(values[2:].index('1')) + '_'
                newName = newName + values[0]
                values.append(newName)
                my_list.append(values)
                ok_count += 1
                #print(newName)
            except:
                newName = newName + str(-1)
                fail_count += 1

print(fail_count)
print(ok_count)

with open('names.csv', 'w+') as f:
    f.write('old_name;new_name\n')
    for file in my_list:
        f.write(file[0] + '.jpg;' + file[-1] + '.jpg\n')


#10005602_654219647977685_1880960612_a.jpg - man/senior
#0a80403abd8b11e2bccd22000ae91234_6.jpg - man/middle age
#927932_699532936769104_1386911032_a.jpg - man/teenager
#10005602_419559608188712_1364292804_a.jpg - man/youth
#10005652_213747508834759_2031083320_a.jpg - man/child
#10009877_469288789865939_1059049450_a.jpg - man/baby

#10013033_267141560113901_1792403080_a.jpg - fem/senior
#10011226_1459133627654519_1075602299_a.jpg - fem/middle age
#10011222_226162354242964_747909117_a.jpg - fem/teenager
#10011222_1513456155548506_1069297718_a.jpg - fem/youth
#10011221_774894735854516_2134162852_a.jpg - fem/child
#10011201_229619030579129_1801187320_a.jpg - fem/baby
