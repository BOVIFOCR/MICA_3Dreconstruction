# -*- coding: utf-8 -*-
# Author: Bernardo Biesseck

import os, sys
import cv2
import numpy as np

import glob
import matplotlib.pyplot as plt
# import imagesize



# LIST OF FOLDERS
if __name__ == '__main__':

    # folders = [
    #     '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463',
    #     '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_1',
    #     '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_5',
    # ]

    folders = [
        
        # # FRGC
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04200',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04201',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04202',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04203',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04211',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04212',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04213',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04214',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04217',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04219',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04221',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04222',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04225',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04226',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04228',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04229',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04233',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04236',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04237',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04239',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04243',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04252',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04256',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04257',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04261',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04265',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04267',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04273',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04274',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04279',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04282',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04284',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04286',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04287',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04288',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04297',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04298',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04299',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04300',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04301',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04302',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04305',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04306',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04308',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04309',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04311',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04312',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04313',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04314',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04315',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04316',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04317',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04319',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04320',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04321',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04322',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04323',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04324',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04327',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04329',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04331',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04334',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04335',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04336',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04337',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04338',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04339',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04341',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04343',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04344',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04347',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04349',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04350',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04351',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04352',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04360',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04361',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04362',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04365',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04366',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04367',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04368',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04369',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04370',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04371',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04372',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04373',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04374',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04378',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04379',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04380',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04381',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04382',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04385',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04386',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04387',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04388',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04391',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04392',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04394',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04395',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04397',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04400',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04402',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04403',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04404',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04406',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04407',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04408',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04409',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04410',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04411',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04412',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04414',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04417',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04418',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04419',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04422',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04423',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04424',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04425',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04427',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04428',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04429',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04430',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04431',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04433',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04434',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04435',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04436',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04437',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04440',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04442',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04444',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04446',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04447',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04448',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04449',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04451',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04453',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04454',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04456',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04460',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04461',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04463',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04464',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04467',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04470',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04471',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04472',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04473',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04475',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04476',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04477',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04478',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04479',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04481',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04482',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04484',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04485',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04487',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04488',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04489',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04493',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04494',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04495',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04496',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04500',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04502',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04503',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04504',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04505',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04506',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04507',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04508',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04509',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04511',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04512',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04513',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04514',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04515',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04519',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04522',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04523',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04524',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04525',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04527',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04529',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04530',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04531',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04533',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04535',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04537',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04538',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04539',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04540',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04542',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04544',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04545',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04546',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04548',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04549',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04553',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04554',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04556',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04557',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04558',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04559',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04560',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04561',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04563',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04568',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04569',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04572',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04575',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04576',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04577',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04578',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04579',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04580',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04581',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04582',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04583',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04584',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04585',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04586',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04587',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04588',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04589',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04590',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04592',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04593',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04595',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04596',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04597',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04598',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04599',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04600',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04601',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04602',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04603',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04604',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04605',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04606',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04607',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04608',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04609',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04610',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04612',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04613',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04615',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04617',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04618',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04619',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04620',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04621',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04622',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04623',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04624',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04626',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04627',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04628',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04629',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04631',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04632',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04633',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04635',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04637',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04638',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04639',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04641',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04642',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04643',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04644',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04645',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04646',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04647',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04650',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04651',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04652',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04653',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04654',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04655',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04656',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04657',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04658',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04659',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04661',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04662',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04663',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04664',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04666',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04667',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04669',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04670',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04671',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04673',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04675',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04676',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04679',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04681',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04682',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04683',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04684',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04686',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04687',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04688',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04690',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04691',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04692',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04693',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04694',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04695',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04696',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04697',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04698',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04699',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04700',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04701',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04702',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04703',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04704',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04705',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04707',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04708',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04709',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04710',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04712',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04714',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04715',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04716',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04717',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04718',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04719',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04720',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04721',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04722',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04724',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04725',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04726',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04727',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04728',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04730',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04731',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04733',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04734',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04735',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04736',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04737',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04738',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04739',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04740',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04742',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04743',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04744',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04745',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04746',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04747',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04748',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04749',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04750',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04751',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04753',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04754',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04755',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04756',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04757',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04758',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04759',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04760',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04761',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04762',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04763',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04764',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04765',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04766',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04767',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04768',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04769',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04770',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04772',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04773',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04774',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04775',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04776',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04777',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04778',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04780',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04782',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04783',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04784',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04785',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04786',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04787',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04788',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04789',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04790',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04791',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04792',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04794',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04795',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04796',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04797',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04798',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04799',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04800',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04801',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04802',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04803',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04804',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04805',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04806',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04807',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04808',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04809',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04810',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04811',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04813',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04814',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04815',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04816',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04817',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04818',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04819',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04820',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04821',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04822',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04823',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04824',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04825',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04826',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04827',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04829',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04830',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04831',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04832',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04833',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04834',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04835',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04836',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04838',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04839',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04840',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04841',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04842',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04843',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04846',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04847',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04848',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04849',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04850',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04851',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04852',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04853',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04854',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04855',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04856',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04857',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04859',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04860',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04861',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04862',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04863',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04864',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04865',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04866',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04867',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04868',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04869',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04870',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04871',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04872',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04873',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04874',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04876',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04877',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04878',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04879',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04880',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04881',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04882',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04883',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04884',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04885',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04886',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04887',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04888',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04889',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04890',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04891',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04892',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04893',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04894',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04896',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04897',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04898',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04899',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04900',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04901',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04902',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04903',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04904',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04905',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04906',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04907',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04908',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04909',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04910',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04911',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04912',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04914',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04915',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04916',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04917',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04918',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04919',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04920',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04921',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04922',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04923',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04924',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04925',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04926',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04927',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04929',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04931',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04932',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04933',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04934',
        # '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/04936',
        

        # # FACEWAREHOUSE
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_1',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_2',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_3',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_4',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_5',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_6',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_7',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_8',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_9',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_10',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_11',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_12',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_13',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_14',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_15',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_16',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_17',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_18',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_19',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_20',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_21',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_22',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_23',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_24',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_25',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_26',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_27',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_28',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_29',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_30',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_31',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_32',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_33',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_34',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_35',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_36',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_37',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_38',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_39',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_40',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_41',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_42',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_43',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_44',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_45',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_46',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_47',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_48',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_49',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_50',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_51',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_52',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_53',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_54',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_55',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_56',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_57',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_58',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_59',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_60',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_61',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_62',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_63',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_64',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_65',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_66',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_67',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_68',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_69',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_70',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_71',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_72',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_73',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_74',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_75',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_76',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_77',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_78',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_79',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_80',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_81',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_82',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_83',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_84',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_85',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_86',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_87',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_88',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_89',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_90',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_91',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_92',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_93',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_94',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_95',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_96',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_97',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_98',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_99',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_100',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_101',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_102',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_103',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_104',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_105',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_106',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_107',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_108',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_109',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_110',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_111',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_112',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_113',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_114',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_115',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_116',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_117',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_118',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_119',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_120',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_121',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_122',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_123',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_124',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_125',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_126',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_127',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_128',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_129',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_130',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_131',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_132',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_133',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_134',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_135',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_136',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_137',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_138',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_139',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_140',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_141',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_142',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_143',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_144',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_145',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_146',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_147',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_148',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_149',
        # '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_150',
        
        
        # STIRLING
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1001',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1002',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1003',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1004',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1005',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1006',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1007',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1008',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1009',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1010',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1011',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1012',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1013',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1014',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1015',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1016',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1017',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1019',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1020',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1021',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1022',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1023',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1024',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1025',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1026',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1027',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1028',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1029',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1030',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1031',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1032',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1034',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1035',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1036',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1037',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1038',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1039',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1040',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1041',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1042',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1043',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1045',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1046',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1047',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1048',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1049',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1050',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1051',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1052',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1053',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1054',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1000',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1001',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1002',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1003',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1004',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1005',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1006',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1007',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1008',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1009',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1010',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1011',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1012',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1013',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1014',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1015',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1016',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1017',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1018',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1019',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1020',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1021',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1022',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1023',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1024',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1025',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1026',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1027',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1028',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1029',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1030',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1031',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1032',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1033',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1034',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1035',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1036',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1037',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1038',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1039',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1040',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1041',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1042',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1043',
        '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/M1044',
        

        # '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00001',
        # '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00002',
        # '/datasets1/bjgbiesseck/MICA/STIRLING/_arcface_input/F1001',
    ]

    img_exts = ['.jpg', '.png']
    arc_img_exts = ['.npy']

    all_img_paths = []
    all_arcface_img_paths = []

    num_rows = len(folders)
    num_cols = 0
    img_size = (0, 0)
    dist_between_rows = 20

    print('Searching files...')
    for i, folder in enumerate(folders):
        print(f'Searching files - folder {i}/{len(folders)-1}: {folder}')
        img_paths = [ glob.glob(folder + '/*' + img_ext) for img_ext in img_exts ]
        arcface_img_paths = [ glob.glob(folder + '/*' + arc_img_ext) for arc_img_ext in arc_img_exts  ]
        # if [] in img_paths:
        #     img_paths.remove([])
        
        img_paths = sorted([ p for path_list in img_paths for p in path_list ])
        arcface_img_paths = sorted([ p for path_list in arcface_img_paths for p in path_list ])
        assert len(img_paths) ==  len(arcface_img_paths)

        if len(img_paths) > num_cols:
            num_cols = len(img_paths) + 1

        # width, height = imagesize.get(img_paths[0])
        # print(f'arcface_img_paths:', arcface_img_paths)
        imsize = np.load(arcface_img_paths[0]).shape
        if imsize[1] > img_size[0] and imsize[2] > img_size[1]:
            img_size = imsize

        all_img_paths.append(img_paths)
        all_arcface_img_paths.append(arcface_img_paths)
        
        # print('img_paths:', img_paths)
        # print('arcface_img_paths:', arcface_img_paths)
        # input('PAUSED')
        # print('-------------')

    print('num_rows:', num_rows)
    print('num_cols:', num_cols)
    print('img_size:', img_size)

    # # OPENCV
    # final_img = np.zeros(shape=(num_rows*img_size[1], num_cols*img_size[2], img_size[0]), dtype=np.float32)
    # print('final_img.shape:', final_img.shape)

    # MATPLOTLIB
    scale_factor = 2.0
    height = scale_factor * num_rows
    width = scale_factor * num_cols
    plt.rcParams["figure.figsize"] = [width, height]
    plt.rcParams["figure.autolayout"] = True
    print('Making initial figure...')
    fig = plt.figure(constrained_layout=False)
    ax_array = fig.subplots(num_rows, num_cols, squeeze=False)
    for row in range(num_rows):
        for col in range(num_cols):
            ax_array[row, col].axis('off')
            # ax_array[row, col].axes.get_xaxis().set_ticks([])
            # ax_array[row, col].axes.get_yaxis().set_ticks([])
            
    print('-----------------------')
    for row in range(num_rows):
        folder, arcface_img_paths = folders[row], all_arcface_img_paths[row]
        dataset_name = folder.split('/')[-3]
        subj_name = folder.split('/')[-1]

        # Add subject name at first column
        # ax_array[row, 0].set_title(dataset_name + '\n' + subj_name, size=7)
        ax_array[row, 0].text(0.5, 0.5, dataset_name+'\n'+subj_name, fontsize=7)

        for col in range(1, len(arcface_img_paths)):
            print('Adding image to figure...')
            print(f'row: {row}/{num_rows-1} - col: {col}/{len(arcface_img_paths)-1}')
            arcface_image_path = arcface_img_paths[col]

            file_name = arcface_image_path.split('/')[-1]   

            print(f'arcface_image_path: {arcface_image_path}')
            arcface_image = np.load(arcface_image_path)
            arcface_image -= arcface_image.min()
            arcface_image /= 2.0
            arcface_image = arcface_image.transpose(1, 2, 0)
            print(f'arcface_image.shape: {arcface_image.shape}')

            # # OPENCV
            # row_start = row*img_size[1]
            # row_end = row*img_size[1]+img_size[1]
            # col_start = col*img_size[2]
            # col_end = col*img_size[2]+img_size[2]
            # arcface_image = cv2.cvtColor(arcface_image, cv2.COLOR_RGB2BGR)
            # final_img[row_start:row_end, col_start:col_end, :] = arcface_image
            # cv2.imshow('final_img', final_img)
            # # cv2.imshow('arcface_image', arcface_image)
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord("q"):
            #     break

            # MATPLOTLIB
            ax_array[row, col].imshow(arcface_image)
            # ax_array[row, col].axes.get_xaxis().set_ticks([])
            # ax_array[row, col].axes.get_yaxis().set_ticks([])
            ax_array[row, col].set_title(file_name, size=7)
            
            print('-----------------------')

    plt.subplots_adjust(left=0.1, right=0.9,
                        bottom=0.1, top=0.9,                        
                        wspace=0.0,
                        hspace=0.6)
    
    file_path = '/home/biesseck/GitHub/ImgViewer_Python_OpenCV/view.png'
    print('Saving figure:', file_path, '...')
    plt.savefig(file_path, bbox_inches="tight", pad_inches = 0)
    # plt.show()

    print('Finished')



# ONE FOLDER
'''
if __name__ == '__main__':
    
    # # FRGC
    # folder = '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463'
    # img_ext = '.jpg'
    # arc_img_ext = '.npy'

    # # FACEWAREHOUSE
    # # folder = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_1'
    # # folder = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_2'
    # # folder = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_5'
    # # folder = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_10'
    # # folder = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_15'
    # # folder = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_16'
    # folder = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_18'
    # img_ext = '.png'
    # arc_img_ext = '.npy'

    # LYHM
    # folder = '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00001'
    # folder = '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00002'
    # folder = '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00005'
    # folder = '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00011'
    folder = '/datasets1/bjgbiesseck/MICA/LYHM/_arcface_input/00013'
    img_ext = '.png'
    arc_img_ext = '.npy'

    img_paths = glob.glob(folder + '/*' + img_ext)
    # arcface_image_path = glob.glob(folder + '/*' + arc_img_ext)

    for image_path in img_paths:
        # image_path =         '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463/02463d170.jpg'
        # arcface_image_path = '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463/02463d170.npy'

        arcface_image_path = image_path.replace(img_ext, arc_img_ext)

        image = cv2.imread(image_path)
        image = image.astype(np.float32) / 255.

        arcface_image = np.load(arcface_image_path)
        # arcface_image -= arcface_image.min()
        arcface_image = arcface_image.transpose(1, 2, 0)
        arcface_image = cv2.cvtColor(arcface_image, cv2.COLOR_RGB2BGR)
        
        image_hist, image_bin_edges     = np.histogram(image, bins=20, range=(-1.0,1.0))
        arcface_hist, arcface_bin_edges = np.histogram(arcface_image, bins=20, range=(-1.0,1.0))
        
        # print('image:', image)
        # print('arcface_image:', arcface_image)

        cv2.imshow('image', image)
        cv2.imshow('arcface_image', arcface_image)
        print('image.shape:', image.shape)
        print('arcface_image.shape:', arcface_image.shape)
        print('image.min():', image.min(), '    image.max():', image.max())
        print('arcface_image.min():', arcface_image.min(), '    arcface_image.max():', arcface_image.max())
        print('arcface_hist:', arcface_hist)
        
        print('---------')

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
'''
            


# ONE IMAGE
'''
if __name__ == '__main__':
    
    # image_path =    '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_1/pose_0.png'
    # arcface_image_path = '/datasets1/bjgbiesseck/MICA/FACEWAREHOUSE/_arcface_input/Tester_1/pose_0.npy'

    image_path =         '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463/02463d170.jpg'
    arcface_image_path = '/datasets1/bjgbiesseck/MICA/FRGC/_arcface_input/02463/02463d170.npy'

    image = cv2.imread(image_path)
    image = image.astype(np.float32) / 255.

    arcface_image = np.load(arcface_image_path)
    arcface_image -= arcface_image.min()
    arcface_image = arcface_image.transpose(1, 2, 0)
    
    
    cv2.imshow('image', image)
    cv2.imshow('arcface_image', arcface_image)
    print('image.shape:', image.shape)
    print('arcface_image:', arcface_image)
    print('arcface_image.shape:', arcface_image.shape)
    cv2.waitKey(0)
'''
