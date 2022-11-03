#!/usr/bin/env python3

import pprint

global counter_r1
global counter_r2
global counter_r3
global counter_r4
counter_r1 = -1
counter_r2 = -1
counter_r3 = -1
counter_r4 = -1

def f1(n):
    i = 1
    r = 0
    counter = 0
    while i <= n:
        counter += 1
        r += i
        i += 1
    return {'r': r, 'counter' : counter}


def f2(n):
    i = 1
    j = 0
    r = 0
    counter = 0
    while i <= n:
        j = 1
        while j <= n:
            counter += 1
            r += 1
            j += 1
        i += 1
    return {'r': r, 'counter' : counter}


def f3(n):
    i = 1
    j = 0
    r = 0
    counter = 0
    while i <= n:
        j = i
        while j <= n:
            counter += 1
            r += 1
            j += 1
        i += 1
    return {'r': r, 'counter' : counter}


def f4(n):
    i = 1
    j = 0
    r = 0
    counter = 0
    while i <= n:
        j = 1
        while j <= i:
            counter += 1
            r += j
            j += 1
        i += 1
    return {'r': r, 'counter' : counter}


def r1(n):
    global counter_r1
    counter_r1 += 1
    if n == 0:
        return 0
    return 1 + r1(n-1)


def r2(n):
    global counter_r2
    counter_r2 += 1
    if n == 0:
        return 0
    if n == 1:
        return 1
    return n + r2(n-2)


def r3(n):
    global counter_r3
    counter_r3 += 1
    if n == 0:
        return 0
    return 1 + 2*r3(n-1)


def r4(n):
    global counter_r4
    counter_r4 += 1
    if n == 0:
        return 0
    return 1 + r4(n-1) + r4(n-1)


def main():
    global counter_r1
    global counter_r2
    global counter_r3
    global counter_r4
    n_max = 20
    global_dict = {'f1': dict.fromkeys(range(n_max)),
                   'f2': dict.fromkeys(range(n_max)),                    
                   'f3': dict.fromkeys(range(n_max)),                    
                   'f4': dict.fromkeys(range(n_max)),                    
                   'r1': dict.fromkeys(range(n_max)),                    
                   'r2': dict.fromkeys(range(n_max)),                    
                   'r3': dict.fromkeys(range(n_max)),                    
                   'r4': dict.fromkeys(range(n_max)),                    
    }

    for n in range(n_max):
        global_dict['f1'][n] = f1(n)
        global_dict['f2'][n] = f2(n)                    
        global_dict['f3'][n] = f3(n)                    
        global_dict['f4'][n] = f4(n)                    
        global_dict['r1'][n] = {'r': r1(n), 'counter' : counter_r1}                   
        global_dict['r2'][n] = {'r': r2(n), 'counter' : counter_r2}                   
        global_dict['r3'][n] = {'r': r3(n), 'counter' : counter_r3}                   
        global_dict['r4'][n] = {'r': r4(n), 'counter' : counter_r4}     
        counter_r1 = -1             
        counter_r2 = -1             
        counter_r3 = -1             
        counter_r4 = -1             

    pprint.pprint(global_dict)

    


if __name__ == "__main__":
    main()