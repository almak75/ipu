#import pandas as DP
import pandas as pd
import math
import numpy as np 
from PIL import Image
import os
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import easyocr
DEVICE = 'cpu'

model1 = YOLO('mod1.pt') #нормальная модель
model2 = YOLO('mod2.pt') #нормальная модель
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

def convert_res_to_pandas_model(results, show_pandas=0):
    boxes = results[0].boxes
    #print(results)
    class_names=results[0].names
    
    if show_pandas:
        print(boxes)
        
    if boxes.shape[0]==0:
        return pd.DataFrame()
    #координаты
    t_xy = boxes.xyxy.cpu()
    #уверенность
    t_conf = boxes.conf.cpu()
    #классы
    t_cls = torch.round(boxes.cls.cpu())
    t_cls = t_cls.to(torch.int)
    df1 = pd.DataFrame(t_xy)
    df2 = pd.DataFrame(t_conf)
    df3 = pd.DataFrame(t_cls)
    df = df1.join(df2,rsuffix='2').join(df3,rsuffix='3')
    df.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence','class']
    df['name'] = df['class'].apply(lambda x:class_names[x])
    return df
    
# получить четверь в которой находится десятичная часть. 
# это нужно обязательно для корректно вращения. 
#
def get_quater(degree, var, dec):
    # var = 1,2 - вдоль какой оси протягивали 1 = х, 2 = у. Протягиваем вдоль той оси к которой ближе повернута прямая, проходящаяя через центры цифр
    #dec  = 0, -1 - после сортировки дробная часть не факт, что окажется внизу. может быть и наверху, тогда счетчик перевернут
    #degree -90...90
    #тут закодированы четверти в зависимости от параметров: знак угла, вариант, где дес точка (в 0 или -1)
    q = {'-/1/-1':1, '-/2/0':1, '+/1/0':2, '+/2/0':2,'-/1/0':3,'-/2/-1':3, '+/2/-1':4,'+/1/-1':4}
    
    
    if degree >=0:
        pos ='+'
    else:
        pos = '-'
    kod =f'{pos}/{var}/{dec}'
    #print(kod)
    if kod in q:
        return q[kod]
    else:
        return 0
        
#эта процедура "двигает счетчик в центр" ии после этого его вращает. Нужно это для того, чтобы при повороте не происходило обрезки счетчика,
#если он расположен скраю изображения

def move_rotation(img,degree, new_center):
    width, height = img.size
    centre_x, centre_y = width//2, height//2
    move_x = int(centre_x - new_center[0])
    move_y = int(centre_y - new_center[1])
    new_image = Image.new('RGB', (width, height), color='black')
    #print('move_x, move_y',move_x, move_y)
    new_image.paste(img, (move_x, move_y))
    #new_image.show()
    return new_image.rotate(degree, center=(centre_x, centre_y),expand=True)
    
    
 #на входе файл с картинкой для обработки
#на выходе: 
#список найденных "повернутых" ипу, к картинке. 
#тут я заморочился. у меня позволяет делать так, чтобы на одной картинке обнаруживать любое количество счетчиков. Не обязательно один. 
#т.е. фоткаем сразу несколько счетчиков и получаем сразу несколько показаний. Просто бомба... и это работатет.

def get_normal_ipu(file, pandas = 0):
    #если pandas = 1, то будут выходить всякие технологические сообщения для отладки
    # Image
    #im = 'D:\\projects\\yolov5\\train_data\\images\\val\\id_1289_value_46_589.png'
    #file = exif_transpose(file)
    im = file
    if pandas==1:
        print(im)
    results = model1([im], verbose = bool(pandas), device=DEVICE)

    
    pd = convert_res_to_pandas_model(results)
    if pandas == 1:
        print('начальное обнаружение. Всё, что нашли:')
        print(pd)
        
    if pd.empty: #не нашли счетчики на картинке
        return None
   
    #ищем центр всех объектов найденных на картинке. Счетчиков может быть несколько, соответственно, цифры надо будет распределить по счетчикам
    pd = pd.assign(centre_x = pd.xmin + (pd.xmax-pd.xmin)/2)
    pd = pd.assign(centre_y = pd.ymin + (pd.ymax-pd.ymin)/2)
    
    #создаемм отдельную таблицу для счетчиков, так как их может быть несколько
    ipu = pd.loc[pd['name']=='ipu']
    if pandas == 1:
        print('найденные счетчики')
        print(ipu)
        
        
    finding_ipu = [] #счетчиков на изображении может быть несколько. здесь будут все найдененые и повернутые.
    
    for index, row in ipu.iterrows():
        if pandas == 1:
            print('\n')
            print('ДАННЫЕ ПО СЧЕТЧИКУ', index+1)
        #это границы счетчика в текущей строке, по ним будем определять точки входящие в счетчик
        x_min, x_max = row['xmin'], row['xmax']
        y_min, y_max = row['ymin'], row['ymax']
        pribor = (x_min, y_min,x_max, y_max)
        
        #фильтруем только те цифры, которые находятся в этом счетчике. Потом возьмем центры и будем строить линию
        temp_ipu = pd.loc[(pd['centre_x'] >= x_min) & (pd['centre_x']<=x_max)  & (pd['centre_y']>=y_min) & (pd['centre_y']<=y_max) & (pd['name'] != 'ipu') & (pd['confidence'] >= 0.4) ]
        if len(temp_ipu) == 0: #в этом счетчике ничего не найдено совсем
            if pandas == 1:
                print('в счетчике нет содержимого. Индекс строки', index)
            continue        
        
        if pandas == 1:
            print('Цифры в счетчике')
            print(temp_ipu)
       
    
        #сначала и ищем вдоль какой оси будем делать это для больше точности, так как тан 90 = это плохо.
        delta_x = temp_ipu['centre_x'].max() - temp_ipu['centre_x'].min()
        delta_y = temp_ipu['centre_y'].max() - temp_ipu['centre_y'].min()
        if delta_x > delta_y:
            var = 1 #больше прижата к Х
            temp_ipu = temp_ipu.sort_values(['centre_x'] )    
        
        else: 
            var = 2 #больше прижата к У
            temp_ipu = temp_ipu.sort_values(['centre_y'] )    
        if pandas == 1:
            print('Вариант', var)
        x = temp_ipu['centre_x'].to_list()
        y = temp_ipu['centre_y'].to_list()
        
        
        if pandas == 1:
            print('Координаты: цифр: ', x,y)
        
        #для дальнейшей работы нам надо минимум 2 цифры, и две координаты. Если меньше, то не повернуть
      
        #print('x1',x)
        #этот кусок нужен, чтоы не взять последню дес цифру, но не известно с какой она стороны
        n = temp_ipu['name'].to_list() 
        if pandas == 1:
            print('Имена классов: ', n)
        dec = ''
        if n[0] == 'dec':
            x = x[1:] #не берем первую точку
            y = y[1:] #не берем первую точку
            dec = 0  #этот флаг потом нужен, чтобы посчитать четверть
        if n[-1] =='dec':
            x = x[:-1] #не берем последнюю точку
            y = y[:-1] #не берем последнюю точку
            dec = -1
        #print('x2',x)
        
        if len(x)<=1:
            if pandas ==1:
                print('Найдено мало цифр. нет данных для поворота')   #это фиаско. какой-то косяк, нот такого ни разу не было. Кстати, если точек всего 2, то polyfit говорит, что результат не надежный
            continue
        #считаем коэф в зависимости от прижатия к оси, чтобы избежать вертикальных вариантов
        #это прям самое гениальное решение. точность сразу выросла.
        if var == 1:
            koef = np.polyfit(np.array(x),np.array(y),  1)
            degree = math.degrees(math.atan(koef[0]))
        else:
            koef = np.polyfit(np.array(y),np.array(x),  1)
            degree = math.degrees(math.atan(koef[0]))
            if degree >= 0:
                degree = 90 - degree
            else:
                degree = -(90 +degree)
        #угол посчитан. это супер он относительно х и показывает направление + или -    
        if pandas == 1:
            print('угол прямой', degree)
        
        q = get_quater(degree, var, dec)#это пипец. Надо определить в какой четверти лежит десятичная часть
        
        if q == 0: #вот это плохая ситуация, но она решаема. это одначает, что в показаниях нет десятичной точки....
            if pandas == 1:
                print('без десятичной части')
            if var == 1:
                t = koef[0]*row['centre_x'] + koef[1]  #определяем где находится прямая в центре счетчика. если значение - меньше центра  все ок. Если значение болььше - счетчик веревернули
                if t >= row['centre_y']:
                    q = 3 #на самом деле это ожмет быть и 4, нно дальше по алгоритму нам все равно
            if var ==2:
                t = koef[0]*row['centre_y'] + koef[1]  #определяем где находится прямая в центре счетчика. если значение - меньше центра  все ок. Если значение болььше - счетчик веревернули
                if t >= row['centre_x']:
                    if degree < 0:
                        degree = 90 +(90 + degree)
                else:
                    if degree > 0:
                        degree = - (180 - degree)
                         
        if pandas == 1:
            print('Четверь', q)
        
        if q == 2 or q == 3:
            degree = 180 + degree
        if pandas == 1:
            print('угол поворота', degree)
        
           
            
        #img = Image.open(im)
        img = im
        #img = exif_transpose(img)
        if pandas:
            print('Вращение вокруг Точки:', row['centre_x'], row['centre_y'])
        #print(img.size)
        #img = img.rotate(degree, center=(row['centre_x'], row['centre_y']),expand=True)
        img = move_rotation(img,degree, new_center=(row['centre_x'], row['centre_y']))
        
        #img = img.rotate(degree, center=(row['centre_x'], row['centre_y']))
        #img = img.rotate(degree,expand=True)
        
        #print(img.size)
        
        #img, centre = rotate_image(im, (row['centre_x'], row['centre_y']), degree)
        #img.show()
        #img.save('Y:\dsf.jpg')
        delta = max(row['xmax']-row['xmin'],row['ymax']-row['ymin'] )//2
        w,h = img.size
        w,h = w//2,h//2
        #img = img.crop((row['centre_x']-delta, row['centre_y']-delta,row['centre_x']+delta, row['centre_y']+delta))
        img = img.crop((w-delta, h-delta, w+delta, h+delta))
        #img.show()
        
        finding_ipu.append(img)
    if pandas == 1:
        print('найдено счетчиков', len(finding_ipu))    
    return finding_ipu   
    
#функция проверяет, попадает ли точка в  прямоугольную область
def in_area(x,y,zone):  #внутри области. попадает ли в прямоугольник
    if math.isnan(zone[0]): #проверяем только если зона есть. 
        return True
    
    if (zone[0] <= x <= zone[2]) and (zone[1] < y < zone[3]):
        return True
    else:
        return False

def in_area_h(x,y,zone): #внутри горизонта. попадает ли в горизонт. Суть в том, что показания находятся в такой горизонтальной области справа и слева от которой никогда ничего нет.
                        #я просто проверяю - попадает ли в эту горизонтальную область.
    if math.isnan(zone[0]):
        return True
    
    if (zone[1] < y < zone[3]):
        return True
    else:
        return False
        
def get_zone(tmp_pd):
    #функция получает на входе пандас.  В нем строки - это найденные прямоегольники цифр.
    #функция ищет минимальный общий прямоугольник, куда входят все прямоугольники на входе.
    #зачем это надо: так я нахожу в каком прямоугольнике находится дробная и в каком целая часть.
    #левый верхний угол
    xmin=tmp_pd['xmin'].min()
    ymin=tmp_pd['ymin'].min()
    #правый нижний
    xmax=tmp_pd['xmax'].max()
    ymax=tmp_pd['ymax'].max()
    return (xmin, ymin, xmax,ymax)
    
    
#сюда поступает файл, в котором уже вырезанный повернутый счетчик
#Задача - найти точку разделения целой и десятичной части или доказать, что ее нет
#на выходе:  границы зон дробной и целой части по отдельности в виде словаря. Потом будем проверять будет входить точка в область или нет
def get_detail_information(file, pandas = 0):
    #если pandas = 1, то будут выходить всякие технологические сообщения для отладки
    # Image
    #im = 'D:\\projects\\yolov5\\train_data\\images\\val\\id_1289_value_46_589.png'
    im = file
    #print(im)

    # Inference
    results = model1([im], verbose = bool(pandas), device=DEVICE)
    pd = convert_res_to_pandas_model(results, pandas)
    if pd.empty: #ничего нет. искать дальне не будем
        return None
    #pd = results.pandas().xyxy[0]
    pd = pd.sort_values(['xmin'] )
    if pandas == 1:
        print('начальное обнаружение для детальной информации')
        print(pd)
    #ищем центр всех объектов найденных на картинке. Счетчиков может быть несколько, соответственно, цифры надо будет распределить по счетчикам
    pd = pd.assign(centre_x = pd.xmin + (pd.xmax-pd.xmin)/2)
    pd = pd.assign(centre_y = pd.ymin + (pd.ymax-pd.ymin)/2)
    
    #создаемм отдельную таблицу для счетчиков, так как их может быть несколько
    pd = pd.loc[pd['name']!='ipu']
    
    сel =pd.loc[pd['name']=='cel']
    dec =pd.loc[pd['name']=='dec']
    
    zone_c = get_zone(сel)
    zone_d = get_zone(dec)
    
    return {'cel':zone_c, 'dec':zone_d}
 
#ОПРЕДЕЛЕНИЕ, ГДЕ НАХОДИТСЯ ЗАПЯТАЯ 
def get_pos_zpt(pd, area_cel, area_dec, show_pandas = 0):
    if show_pandas == 1:
        print('Функция get_pos_zpt. на входе:')
        print('pd',pd)
        print('area_cel',area_cel)
        print('area_dec',area_dec)
        
    #Надо найти Х слева от которого - целая часть, справа - дробная.
    #принцип такой: в идеале у нас три точки: правая часть целой части, запятая и левая часть дробной части.
    #целая части будет всегда. И она почти всегда хорошо находится. ЗПТ - есть не всегда. Дробная часть есть не всегда. Наиборее надежно - найти запятую. это будет победа. Но ее ищет плохо.
    #если нет запятой в пандасе, то дудем брать крайнюю правую границу зоны cel. можно взять левую границу от dec. 
    #в идеале - найдем среднее между тремя точками. Но могут быть выбросы, особенно у дробной части. Да и у целой. Выбросы надо удалить. но точек всего триииии......
    all_pos_zpt = [] #все потенциальные координаты разделителя
    
   
    #Правая нижняя часть целой части, 
    if  not math.isnan(area_cel[2]):
        all_pos_zpt.append(area_cel[2]) #возвращаем х правого нижнего ушла целой части ....надеюсь, что она будет всегда, хоть в каком то виде. Хер! не всегда. Надо исправить!
    #левая часть дробной части
    if not math.isnan(area_dec[0]):
        all_pos_zpt.append(area_dec[0]) #левый верхний Х десятичной части 
    else: 
        all_pos_zpt.append(10000) #десятичной части нет. добавим большое число, чтобы все относилось потом к целому
    #запятая, если нашли
    tmp_zpt = pd.loc[pd['name']=='zpt']    #оставим только найденную zpt
    if not tmp_zpt.empty:
        all_pos_zpt.append(tmp_zpt.iloc[0, tmp_zpt.columns.get_loc('centre_x')]) #Берем из первой строки. Надеюсь она тут одна, но можно бы отсортировать и взять самую уверенную
            
    if show_pandas == 1:
        print('все позиции разделителя!!!!!!!!!!!!!!!!!!!!', all_pos_zpt)
    #есть такая проблема. десятичная часть находится плохо. Иногда она уезжает налево. Это дикий выброс, который нам не нужен 
    #надо его отфильтровать. будем фильтровать не матически, а анализируя расстояние между центрами найденных цифр
   
    #а теперь, если у нас три потенциальные точки - смотрим, не является ли одна из них слишком далекой от двух других. Если дальше, чем среднее расстояние между цифрами, то ее удаляем.
        tmp_digit = pd.loc[pd['name']!='zpt']
        x_pos = tmp_digit['centre_x'].to_list()  #это центры всех цифр. 
        x_pos = [x_pos[i+1] - x_pos[i] for i in range(len(x_pos) - 1)]  #это расстояние всех цифр между центрами. Ширину считать не совсем верно. 
        print('x_pos',x_pos)
        if len(x_pos) == 0: # цифнры не найдены, запятая не имеет смысла
            return all_pos_zpt[-1] #вернем просто позицию запятой
        mean_dist = sum(x_pos) / len(x_pos)   #это среднее расстояние между цифрами. Это была одна из крутых правильных идей. Но надо еще думать.
        if show_pandas == 1:
            print('средняя дистанция между цифрами', mean_dist)
        
        all_pos_zpt.sort() #здесь может быть 1.2.3 числа. мы можем пожертвовать одним или двумя. Но одно точно надо оставить
        #похоже на костыль. надо по другому переписать...но работает для трех чисел отлично.
        if all_pos_zpt[2] -all_pos_zpt[1]> mean_dist*1:  #коэффициент - здесь мы можем расстояние слегка растянуть, но и без растяжения работает норм.
            all_pos_zpt.pop(2)
        if all_pos_zpt[1] -all_pos_zpt[0]> mean_dist*1:
            all_pos_zpt.pop(0)
         
    if show_pandas == 1:
        print('позиции разделителя без выброса', all_pos_zpt)
    
    pos = sum(all_pos_zpt) / len(all_pos_zpt)
    return pos
       
       
#Функция возвращает показания найденные на счетчике. Это самый центр вселенной
def get_nomer_easyocr_ready_img(img, rect):
    #print('Для вырезки номера',pic, rect)
    #img = Image.open(pic)
    img = img.crop(rect)
    w,h = img.size
    
    if w < h: #номер вертикальны
        img = img.rotate(-90, expand=True)
        img = PIL.ImageOps.invert(img)
    
    
    #img.show()
    h_new =  64

    delta = h_new/h
    w_new = int(delta*w)
    img = img.resize((w_new,h_new),Image.NEAREST) #PIL.Image.NEAREST  PIL.Image.BILINEAR  PIL.Image.BICUBIC PIL.Image.LANCZOS
    #img.show()
    result = reader.readtext(np.array(img), decoder = 'beamsearch',blocklist='()!@#$%^&*-',detail = 0, min_size =15,allowlist =' -1234567890') #
    #print(result)
    return result

def dust(st):
    st = st.replace(' ','')
    st = st.replace('NO','')
    st = st.replace('№','')
    st = st.replace('N°','')
    st = st.replace('N','')
    st = st.replace('O','0')
    st = st.replace(',','.')
    st = st.replace('--','-')
    st = st.replace('-','')
    st = st.replace('.','')
    return st
    
def get_digits(file, show_pandas = 0,  digit_area=''):                                       #функия возвращает цифры, распознанные на счетчике
    #digit_area  - это область цифр. отдельно целых и отдельно десятичных
    im = file
    results = model2([im], verbose = bool(show_pandas), device=DEVICE)
#     if show_pandas:
#             print('Нашли тензор')
#             print(results)
    pd = convert_res_to_pandas_model(results,show_pandas )
    if pd.empty: #не нашли счетчики на картинке
        if show_pandas:
            print('не нашли счетчика на картинке.')
        return None,None
   
    pd = pd.assign(centre_x = pd.xmin + (pd.xmax-pd.xmin)/2)
    pd = pd.assign(centre_y = pd.ymin + (pd.ymax-pd.ymin)/2)
    
    pd = pd.sort_values(['centre_x'] )                          #сортируем найденные цифры по порядку. В идеальном мире - в колонке name Будут уже видны показания.
    
    
    if show_pandas == 1:
        print('Детекция цифр, все, что нашлось')
        print(pd)
    
    #работаем с заводским номером #######################################################################################################################################################
    #######################################################################################################################################################
    #######################################################################################################################################################
    nomer = None
    pd_nomer = pd.loc[pd['name'] == 'nomer']
    if show_pandas:
        print('Нашли заводской номер',pd_nomer)    
    #прямоугольник для вырезки номера
    if not pd_nomer.empty:
        nomer_x_min = pd_nomer.iloc[0, pd_nomer.columns.get_loc('xmin')]
        nomer_y_min = pd_nomer.iloc[0, pd_nomer.columns.get_loc('ymin')]
        nomer_x_max = pd_nomer.iloc[0, pd_nomer.columns.get_loc('xmax')]
        nomer_y_max = pd_nomer.iloc[0, pd_nomer.columns.get_loc('ymax')]
        nomer = get_nomer_easyocr_ready_img(file, (nomer_x_min, nomer_y_min, nomer_x_max,nomer_y_max))
        if show_pandas:
            print('Номер в прямоугольнике',(nomer_x_min, nomer_y_min, nomer_x_max,nomer_y_max))    

        nomer =''.join(list(nomer)).upper().strip()
        nomer = dust(nomer)
    #######################################################################################################################################################
    #######################################################################################################################################################
    #######################################################################################################################################################
    #######################################################################################################################################################
    
    
    
    pd = pd.loc[pd['name'] != 'nomer']                      #Убираем лишнее. заводской номер нам пока не нужен
    
    pd = pd.loc[((pd.confidence > 0.5)  | (pd.name == 'zpt'))]  #оставляем только уверенные цифры и запятую. Запятая нужна в любом случае
    #попробуем сделать так, если цифр нашлось мало, снизим уверенность
    if pd.shape[0]<=4:
        pd = pd.loc[((pd.confidence > 0.3)  | (pd.name == 'zpt'))]  #оставляем только уверенные цифры и запятую. Запятая нужна в любом случае
    
    
    #==================================================================================================================================
    #была такая ситуация, когда на счетчике находилось НЕСКОЛЬКО ЗАПЯТЫХ из за грязи на табло. id_126_value_769_703.jpg
    #надо сделать так, чтобы осталась только одна запятая с максимальной степенью уверенности
    zpt = pd.loc[(pd.name == 'zpt')]
    max_confidence_zpt = zpt['confidence'].max()
    #Оставляем только  zpt с нужным нам максимальным значением уверенности
    pd = pd.loc[ ((pd.name != 'zpt') | (pd.confidence == max_confidence_zpt))]
    #после отсеивания лишнего zpt 
    
    #==================================================================================================================================
    #отрабатываем еще одну ситуацию. иногда сетка находит хрен знает какие цифры, которые находятся вообще сильно выше или сильно ниже, например в номере или в названии счетчика. 
    #Такие цифры надо отсеитьи убрать из таблицы
    #принцип такой - берем цифру - если она не попадает в горизонт. Пример такого файла  id_209_value_107_926.jpg
    #тут проблема с запятой. не видит ее.id_232_value_767_784.jpg
    #вот тут косяк с запятой id_180_value_233_116.jpg

    for index, row in pd.iterrows():
        if not in_area_h(row['centre_x'],row['centre_y'],digit_area['cel']) and not in_area(row['centre_x'],row['centre_y'],digit_area['dec']):  #не в области целых чисел и не в области дробной части, тогда в топку...
            if row['name']!='zpt': #запятую не будем удалять на всякий случай
                pd = pd.drop(index=index)
  

    if show_pandas == 1:
        print('то, что попало в область')
        print(pd)
    
    #==================================================================================================================================
    #и еще одна ситуация. Она возникает - когда в одном разряде торчат 2 цифры друг над другом. Сетка находит обе. Оставить надо одну...
    #оставлять будем ту, уверенность в которой больше. Но надо найти те цифры - которыеы сильно пересекаются по х. Для этого будем искать % перекрытия.
    # пример id_98_value_304_465.jpg, id_3_value_366_885.jpg
    
    #фиг. оставлять надо не ту цифру, в которой уверенность больше. Надо попробовать оставить: 
    #или большую цифр
    #или цифру центр которых ближе к оси предыдущих цифр.  Это жестко? ну да, но похоже, что уверенность - брать не надо.
    #koef = np.polyfit(np.array(x), np.array(y), 1)
    #вариант 1
    vrem = []
    for index, row in pd.iterrows():
        #print('fgdfgdfg',row['ymin'])
        if row['name'] =='zpt':
            continue
        #vrem.append([row['xmin'],row['xmax'],row['confidence'], index]) #собираем левую и правую границу каждой цифры
        vrem.append({'xmin': row['xmin'],'xmax': row['xmax'], 'confidence': row['confidence'], 'index': index, 'centre_x': row['centre_x'], 'centre_y':row['centre_y'],'ymin': row['ymin'],'ymax': row['ymax']}) #собираем левую и правую границу каждой цифры
    
    if len(vrem)>1: #иначе смысла нет
        #они будут содержать список уверенных точек по которым можнно будет построить ось цифр. в случае спорных моментов - мы будем брать ту цифру. которая ближе к оси
        #список будут попадать только те цифры, которые "уверено" одни в разряде
        
        #но здесь возможны два варианта. В одном разряде нейронка может найти 
        #1. две цифры друг под другом. В этом случае мы берем ту, которая ближе к оси цифр. Ось ищется по цифрам других разрядов
        #2. он одну цифру узнает как например 5 или 6. т.е. в одном месте находит сразу две цифры. В этом случае, надо брать ту цифру, уверенность в которой большеэ
        
        good_x = []
        good_y = []
        double = [] #сюда сложим номера цифр, где есть 2 или более кандидатов на одно место в показаниях

        for i in range(0, len(vrem)-1):
            dl = vrem[i]['xmax'] -vrem[i]['xmin'] #это длина текущего промежутка
            #dl = vrem[i][1] -vrem[i][0] #это длина текущего промежутка
            common = min(vrem[i]['xmax'], vrem[i+1]['xmax']) - max(vrem[i]['xmin'], vrem[i+1]['xmin']) #Общая часть со следующей цифрой
            #common = min(vrem[i][1], vrem[i+1][1]) - max(vrem[i][0], vrem[i+1][0]) #Общая часть со следующей цифрой
            
            common_pr = common/dl #процент совпадения
            if common_pr < 0.5: #значит перекрытие довольно существенное. Удалим ту строку, у которой меньше уверенность
                #если предыдующая метка надежная
                if i-1 not in double:
                    good_x.append(vrem[i]['centre_x'])
                    good_y.append(vrem[i]['centre_y']) 
            if common_pr > 0.75: #есть перекрытие больше 75%, эту цифру надо будет потом обработать, чтобы выяснить - что брать
                double.append(i)
        
        #Теперь проверяем двойников, если они там есть
        if len(double) > 0:
            koef = np.polyfit(np.array(good_x), np.array(good_y), 1)
            for i in double:  #i и следующая за ней цифра - кандидаты на одно место а возьмем того, который ближе к нашей оси
                #надо рассмотреть 2 варианта. Цифры друг под другом или друг на друге
                min_bound = min(vrem[i]['ymax'],vrem[i+1]['ymax'])
                if ((vrem[i]['centre_y'] < min_bound) and (vrem[i+1]['centre_y'] < min_bound)):    #оба центра находятся ниже минимальной границы двух изобра - значит совпадают
                    #берем ту, в которой уверенности больше
                    if vrem[i]['confidence'] > vrem[i+1]['confidence']:
                        deleting = vrem[i+1]['index']
                    else:
                        deleting = vrem[i]['index']
                    try:   #может уже ранее ее удалили
                        pd = pd.drop(index=deleting)
                    except:
                        pass
            
                else:
                    delta1 = abs(vrem[i]['centre_y']   - (koef[0] * vrem[i]['centre_x'] + koef[1])) 
                    delta2 = abs(vrem[i+1]['centre_y']   - (koef[0] * vrem[i+1]['centre_x'] + koef[1]))
                    #удаляем ту цифру, дельта которой больше, т.е. она остоит дальше от прямой
                    if delta1 > delta2:
                        deleting = vrem[i]['index']
                    else:
                        deleting = vrem[i+1]['index']

                    try:   #может уже ранее ее удалили
                        pd = pd.drop(index=deleting)
                    except:
                        pass
    #по сути на этот момент должны остаться только верные цифры и запятая в пандасе.
   
    d,c = '', ''
    pos_zpt = get_pos_zpt(pd, digit_area['cel'], digit_area['dec'],show_pandas)   #бомбическая функция вставки запятой.
    if show_pandas == 1:
        print('позиция запятой', pos_zpt)
    for index, row in pd.iterrows():
        #print('fgdfgdfg',row['ymin'])
        if row['name'] =='zpt':
            continue        
        if row['centre_x'] <= pos_zpt: #попадает в зону целых чисел.
            c +=row['name']
        elif row['centre_x'] >= pos_zpt: #попадает в зону дробной части
            d +=row['name']
    if not c and not d:
        return None, None

    d=f'{c}.{d}'
    return d, nomer
    
def look_to_file(im): 
    #основная процедура получения показаний из файлы
    im_rez = get_normal_ipu(im,0) #сначала получаем список повернутых изображений
    rez = []
    if im_rez:
        for img in im_rez:
            #img.show()
            detail_info = get_detail_information(img,0)
            #print(detail_info)
            if detail_info:
                pokaz, z_nomer = get_digits(img, digit_area = detail_info, show_pandas=0)
                rez.append({'pokaz':pokaz,'z_nomer':z_nomer}) #собираем показания и заводской номер
    return rez
    
def my_test(im):
    return 1