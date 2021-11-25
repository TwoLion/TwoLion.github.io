test
================

## dplyr

-   Tidyverse의 한 library로, dataframe 정리에 사용
-   main function
    -   select
    -   filter
    -   arrange
    -   mutate
    -   summarise
    -   group\_by
-   사용할 데이터로, nycflights13 사용

## 데이터 불러오기, library setting

``` r
library(nycflights13)
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.0 --

    ## v ggplot2 3.3.3     v purrr   0.3.4
    ## v tibble  3.1.0     v dplyr   1.0.5
    ## v tidyr   1.1.3     v stringr 1.4.0
    ## v readr   1.4.0     v forcats 0.5.1

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
print(flights)
```

    ## # A tibble: 336,776 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      517            515         2      830            819
    ##  2  2013     1     1      533            529         4      850            830
    ##  3  2013     1     1      542            540         2      923            850
    ##  4  2013     1     1      544            545        -1     1004           1022
    ##  5  2013     1     1      554            600        -6      812            837
    ##  6  2013     1     1      554            558        -4      740            728
    ##  7  2013     1     1      555            600        -5      913            854
    ##  8  2013     1     1      557            600        -3      709            723
    ##  9  2013     1     1      557            600        -3      838            846
    ## 10  2013     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

## 데이터 확인하기

``` r
print(summary(flights))
```

    ##       year          month             day           dep_time    sched_dep_time
    ##  Min.   :2013   Min.   : 1.000   Min.   : 1.00   Min.   :   1   Min.   : 106  
    ##  1st Qu.:2013   1st Qu.: 4.000   1st Qu.: 8.00   1st Qu.: 907   1st Qu.: 906  
    ##  Median :2013   Median : 7.000   Median :16.00   Median :1401   Median :1359  
    ##  Mean   :2013   Mean   : 6.549   Mean   :15.71   Mean   :1349   Mean   :1344  
    ##  3rd Qu.:2013   3rd Qu.:10.000   3rd Qu.:23.00   3rd Qu.:1744   3rd Qu.:1729  
    ##  Max.   :2013   Max.   :12.000   Max.   :31.00   Max.   :2400   Max.   :2359  
    ##                                                  NA's   :8255                 
    ##    dep_delay          arr_time    sched_arr_time   arr_delay       
    ##  Min.   : -43.00   Min.   :   1   Min.   :   1   Min.   : -86.000  
    ##  1st Qu.:  -5.00   1st Qu.:1104   1st Qu.:1124   1st Qu.: -17.000  
    ##  Median :  -2.00   Median :1535   Median :1556   Median :  -5.000  
    ##  Mean   :  12.64   Mean   :1502   Mean   :1536   Mean   :   6.895  
    ##  3rd Qu.:  11.00   3rd Qu.:1940   3rd Qu.:1945   3rd Qu.:  14.000  
    ##  Max.   :1301.00   Max.   :2400   Max.   :2359   Max.   :1272.000  
    ##  NA's   :8255      NA's   :8713                  NA's   :9430      
    ##    carrier              flight       tailnum             origin         
    ##  Length:336776      Min.   :   1   Length:336776      Length:336776     
    ##  Class :character   1st Qu.: 553   Class :character   Class :character  
    ##  Mode  :character   Median :1496   Mode  :character   Mode  :character  
    ##                     Mean   :1972                                        
    ##                     3rd Qu.:3465                                        
    ##                     Max.   :8500                                        
    ##                                                                         
    ##      dest              air_time        distance         hour      
    ##  Length:336776      Min.   : 20.0   Min.   :  17   Min.   : 1.00  
    ##  Class :character   1st Qu.: 82.0   1st Qu.: 502   1st Qu.: 9.00  
    ##  Mode  :character   Median :129.0   Median : 872   Median :13.00  
    ##                     Mean   :150.7   Mean   :1040   Mean   :13.18  
    ##                     3rd Qu.:192.0   3rd Qu.:1389   3rd Qu.:17.00  
    ##                     Max.   :695.0   Max.   :4983   Max.   :23.00  
    ##                     NA's   :9430                                  
    ##      minute        time_hour                  
    ##  Min.   : 0.00   Min.   :2013-01-01 05:00:00  
    ##  1st Qu.: 8.00   1st Qu.:2013-04-04 13:00:00  
    ##  Median :29.00   Median :2013-07-03 10:00:00  
    ##  Mean   :26.23   Mean   :2013-07-03 05:22:54  
    ##  3rd Qu.:44.00   3rd Qu.:2013-10-01 07:00:00  
    ##  Max.   :59.00   Max.   :2013-12-31 23:00:00  
    ## 

## Select

특정 열을 선택해주는 함수

용법 : select(dataframe, columns(벡터로 입력해도 되고, 하나의 값으로
입력가능하고, :이나 - 사용 가능))

``` r
select(flights, year, month, day)
```

    ## # A tibble: 336,776 x 3
    ##     year month   day
    ##    <int> <int> <int>
    ##  1  2013     1     1
    ##  2  2013     1     1
    ##  3  2013     1     1
    ##  4  2013     1     1
    ##  5  2013     1     1
    ##  6  2013     1     1
    ##  7  2013     1     1
    ##  8  2013     1     1
    ##  9  2013     1     1
    ## 10  2013     1     1
    ## # ... with 336,766 more rows

``` r
select(flights, c(year, month, day))
```

    ## # A tibble: 336,776 x 3
    ##     year month   day
    ##    <int> <int> <int>
    ##  1  2013     1     1
    ##  2  2013     1     1
    ##  3  2013     1     1
    ##  4  2013     1     1
    ##  5  2013     1     1
    ##  6  2013     1     1
    ##  7  2013     1     1
    ##  8  2013     1     1
    ##  9  2013     1     1
    ## 10  2013     1     1
    ## # ... with 336,766 more rows

``` r
select(flights, year:day)
```

    ## # A tibble: 336,776 x 3
    ##     year month   day
    ##    <int> <int> <int>
    ##  1  2013     1     1
    ##  2  2013     1     1
    ##  3  2013     1     1
    ##  4  2013     1     1
    ##  5  2013     1     1
    ##  6  2013     1     1
    ##  7  2013     1     1
    ##  8  2013     1     1
    ##  9  2013     1     1
    ## 10  2013     1     1
    ## # ... with 336,766 more rows

``` r
select(flights, -year)
```

    ## # A tibble: 336,776 x 18
    ##    month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1     1     1      517            515         2      830            819
    ##  2     1     1      533            529         4      850            830
    ##  3     1     1      542            540         2      923            850
    ##  4     1     1      544            545        -1     1004           1022
    ##  5     1     1      554            600        -6      812            837
    ##  6     1     1      554            558        -4      740            728
    ##  7     1     1      555            600        -5      913            854
    ##  8     1     1      557            600        -3      709            723
    ##  9     1     1      557            600        -3      838            846
    ## 10     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

### 특정 문자열 찾아내기

-   starts\_with(‘문자’) : ’문자’로 시작하는 단어 모두 찾기
-   ends\_with(‘문자’) : ’문자’로 끝나는 단어 모두 찾기
-   matches(‘문자’) : ’문자’를 포함하는 단어 모두 찾기
-   contains(‘문자’) : ’문자’를 포함하는 단어 모두 찾기

``` r
select(flights, starts_with('dep'))
```

    ## # A tibble: 336,776 x 2
    ##    dep_time dep_delay
    ##       <int>     <dbl>
    ##  1      517         2
    ##  2      533         4
    ##  3      542         2
    ##  4      544        -1
    ##  5      554        -6
    ##  6      554        -4
    ##  7      555        -5
    ##  8      557        -3
    ##  9      557        -3
    ## 10      558        -2
    ## # ... with 336,766 more rows

``` r
select(flights, ends_with('delay'))
```

    ## # A tibble: 336,776 x 2
    ##    dep_delay arr_delay
    ##        <dbl>     <dbl>
    ##  1         2        11
    ##  2         4        20
    ##  3         2        33
    ##  4        -1       -18
    ##  5        -6       -25
    ##  6        -4        12
    ##  7        -5        19
    ##  8        -3       -14
    ##  9        -3        -8
    ## 10        -2         8
    ## # ... with 336,766 more rows

``` r
select(flights, matches('time'))
```

    ## # A tibble: 336,776 x 6
    ##    dep_time sched_dep_time arr_time sched_arr_time air_time time_hour          
    ##       <int>          <int>    <int>          <int>    <dbl> <dttm>             
    ##  1      517            515      830            819      227 2013-01-01 05:00:00
    ##  2      533            529      850            830      227 2013-01-01 05:00:00
    ##  3      542            540      923            850      160 2013-01-01 05:00:00
    ##  4      544            545     1004           1022      183 2013-01-01 05:00:00
    ##  5      554            600      812            837      116 2013-01-01 06:00:00
    ##  6      554            558      740            728      150 2013-01-01 05:00:00
    ##  7      555            600      913            854      158 2013-01-01 06:00:00
    ##  8      557            600      709            723       53 2013-01-01 06:00:00
    ##  9      557            600      838            846      140 2013-01-01 06:00:00
    ## 10      558            600      753            745      138 2013-01-01 06:00:00
    ## # ... with 336,766 more rows

``` r
select(flights, contains('time'))
```

    ## # A tibble: 336,776 x 6
    ##    dep_time sched_dep_time arr_time sched_arr_time air_time time_hour          
    ##       <int>          <int>    <int>          <int>    <dbl> <dttm>             
    ##  1      517            515      830            819      227 2013-01-01 05:00:00
    ##  2      533            529      850            830      227 2013-01-01 05:00:00
    ##  3      542            540      923            850      160 2013-01-01 05:00:00
    ##  4      544            545     1004           1022      183 2013-01-01 05:00:00
    ##  5      554            600      812            837      116 2013-01-01 06:00:00
    ##  6      554            558      740            728      150 2013-01-01 05:00:00
    ##  7      555            600      913            854      158 2013-01-01 06:00:00
    ##  8      557            600      709            723       53 2013-01-01 06:00:00
    ##  9      557            600      838            846      140 2013-01-01 06:00:00
    ## 10      558            600      753            745      138 2013-01-01 06:00:00
    ## # ... with 336,766 more rows

## Filter

특정 행을 선택해주는 함수

용법 : fliter(dataframe, 조건)

``` r
filter(flights, carrier=='US')
```

    ## # A tibble: 20,536 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      622            630        -8     1017           1014
    ##  2  2013     1     1      627            630        -3     1018           1018
    ##  3  2013     1     1      629            630        -1      824            833
    ##  4  2013     1     1      643            645        -2      837            848
    ##  5  2013     1     1      752            759        -7      955            959
    ##  6  2013     1     1      811            815        -4     1026           1016
    ##  7  2013     1     1      823            825        -2     1019           1024
    ##  8  2013     1     1      908            915        -7     1004           1033
    ##  9  2013     1     1      955           1000        -5     1336           1325
    ## 10  2013     1     1      959           1000        -1     1151           1206
    ## # ... with 20,526 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

``` r
filter(flights, month>=3, month<=5)
```

    ## # A tibble: 85,960 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     3     1        4           2159       125      318             56
    ##  2  2013     3     1       50           2358        52      526            438
    ##  3  2013     3     1      117           2245       152      223           2354
    ##  4  2013     3     1      454            500        -6      633            648
    ##  5  2013     3     1      505            515       -10      746            810
    ##  6  2013     3     1      521            530        -9      813            827
    ##  7  2013     3     1      537            540        -3      856            850
    ##  8  2013     3     1      541            545        -4     1014           1023
    ##  9  2013     3     1      549            600       -11      639            703
    ## 10  2013     3     1      550            600       -10      747            801
    ## # ... with 85,950 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

``` r
filter(flights, month>=3 & month<=5)
```

    ## # A tibble: 85,960 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     3     1        4           2159       125      318             56
    ##  2  2013     3     1       50           2358        52      526            438
    ##  3  2013     3     1      117           2245       152      223           2354
    ##  4  2013     3     1      454            500        -6      633            648
    ##  5  2013     3     1      505            515       -10      746            810
    ##  6  2013     3     1      521            530        -9      813            827
    ##  7  2013     3     1      537            540        -3      856            850
    ##  8  2013     3     1      541            545        -4     1014           1023
    ##  9  2013     3     1      549            600       -11      639            703
    ## 10  2013     3     1      550            600       -10      747            801
    ## # ... with 85,950 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

## Arrange

dataframe의 배열을 바꿔주는 함수

용법 : arrange(dataframe, column1, column2, … )

해당하는 column 순서대로 오름차순 or 내림차순으로 정렬해줌(default :
오름차순)

내림차순 하는 방법 : desc(columns)

``` r
# year에 대해 오름차순으로 정렬

arrange(flights, year)
```

    ## # A tibble: 336,776 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      517            515         2      830            819
    ##  2  2013     1     1      533            529         4      850            830
    ##  3  2013     1     1      542            540         2      923            850
    ##  4  2013     1     1      544            545        -1     1004           1022
    ##  5  2013     1     1      554            600        -6      812            837
    ##  6  2013     1     1      554            558        -4      740            728
    ##  7  2013     1     1      555            600        -5      913            854
    ##  8  2013     1     1      557            600        -3      709            723
    ##  9  2013     1     1      557            600        -3      838            846
    ## 10  2013     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

``` r
# year에 대해 오름차순으로 정렬하고, month에 대해 오름차순으로 정렬한 후, day에 대해 오름차순으로 정렬

arrange(flights, year, month, day)
```

    ## # A tibble: 336,776 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      517            515         2      830            819
    ##  2  2013     1     1      533            529         4      850            830
    ##  3  2013     1     1      542            540         2      923            850
    ##  4  2013     1     1      544            545        -1     1004           1022
    ##  5  2013     1     1      554            600        -6      812            837
    ##  6  2013     1     1      554            558        -4      740            728
    ##  7  2013     1     1      555            600        -5      913            854
    ##  8  2013     1     1      557            600        -3      709            723
    ##  9  2013     1     1      557            600        -3      838            846
    ## 10  2013     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

``` r
# year에 대해 오름차순으로 정렬하고, month에 대해 내림차순으로 정렬한 후, day에 대해 오름차순으로 정렬

arrange(flights, year, desc(month), day)
```

    ## # A tibble: 336,776 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013    12     1       13           2359        14      446            445
    ##  2  2013    12     1       17           2359        18      443            437
    ##  3  2013    12     1      453            500        -7      636            651
    ##  4  2013    12     1      520            515         5      749            808
    ##  5  2013    12     1      536            540        -4      845            850
    ##  6  2013    12     1      540            550       -10     1005           1027
    ##  7  2013    12     1      541            545        -4      734            755
    ##  8  2013    12     1      546            545         1      826            835
    ##  9  2013    12     1      549            600       -11      648            659
    ## 10  2013    12     1      550            600       -10      825            854
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

## Mutate

dataframe의 열을 추가 or 바꿔주는 함수

용법 : mutate(dataframe, column\_name = 조건)

만약 새로운 column\_name을 사용하였다면, dataframe에 새로운 column이
추가 됨

만약 기존에 있던 column\_name을 사용하였다면, 원래 있던 column의
데이터가 변화됨.

``` r
mutate(flights, new_time = hour+(minute/60))
```

    ## # A tibble: 336,776 x 20
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      517            515         2      830            819
    ##  2  2013     1     1      533            529         4      850            830
    ##  3  2013     1     1      542            540         2      923            850
    ##  4  2013     1     1      544            545        -1     1004           1022
    ##  5  2013     1     1      554            600        -6      812            837
    ##  6  2013     1     1      554            558        -4      740            728
    ##  7  2013     1     1      555            600        -5      913            854
    ##  8  2013     1     1      557            600        -3      709            723
    ##  9  2013     1     1      557            600        -3      838            846
    ## 10  2013     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 12 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>,
    ## #   new_time <dbl>

``` r
mutate(flights, hour = hour+(minute/60))
```

    ## # A tibble: 336,776 x 19
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      517            515         2      830            819
    ##  2  2013     1     1      533            529         4      850            830
    ##  3  2013     1     1      542            540         2      923            850
    ##  4  2013     1     1      544            545        -1     1004           1022
    ##  5  2013     1     1      554            600        -6      812            837
    ##  6  2013     1     1      554            558        -4      740            728
    ##  7  2013     1     1      555            600        -5      913            854
    ##  8  2013     1     1      557            600        -3      709            723
    ##  9  2013     1     1      557            600        -3      838            846
    ## 10  2013     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

### transmute

mutate할 열만 보여주는 함수

``` r
transmute(flights, new_time = hour+(minute/60))
```

    ## # A tibble: 336,776 x 1
    ##    new_time
    ##       <dbl>
    ##  1     5.25
    ##  2     5.48
    ##  3     5.67
    ##  4     5.75
    ##  5     6   
    ##  6     5.97
    ##  7     6   
    ##  8     6   
    ##  9     6   
    ## 10     6   
    ## # ... with 336,766 more rows

``` r
transmute(flights, hour = hour+(minute/60))
```

    ## # A tibble: 336,776 x 1
    ##     hour
    ##    <dbl>
    ##  1  5.25
    ##  2  5.48
    ##  3  5.67
    ##  4  5.75
    ##  5  6   
    ##  6  5.97
    ##  7  6   
    ##  8  6   
    ##  9  6   
    ## 10  6   
    ## # ... with 336,766 more rows

## Summarise

dataframe의 열의 특징을 요약해주는 함수

용법 : summarise(dataframe, name = function results)

``` r
summarise(flights, dep_delay=mean(dep_delay, na.rm=T), arr_delay=mean(arr_delay, na.rm=T))
```

    ## # A tibble: 1 x 2
    ##   dep_delay arr_delay
    ##       <dbl>     <dbl>
    ## 1      12.6      6.90

## Group\_by

dataframe의 행들을 group화 시켜주는 함수

용법 : group\_by(dataframe, column)

해당 column에 속하는 성분을 이용하여 grouping을 진행

``` r
group_by(flights, origin)
```

    ## # A tibble: 336,776 x 19
    ## # Groups:   origin [3]
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      517            515         2      830            819
    ##  2  2013     1     1      533            529         4      850            830
    ##  3  2013     1     1      542            540         2      923            850
    ##  4  2013     1     1      544            545        -1     1004           1022
    ##  5  2013     1     1      554            600        -6      812            837
    ##  6  2013     1     1      554            558        -4      740            728
    ##  7  2013     1     1      555            600        -5      913            854
    ##  8  2013     1     1      557            600        -3      709            723
    ##  9  2013     1     1      557            600        -3      838            846
    ## 10  2013     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

``` r
group_by(flights, origin, tailnum)
```

    ## # A tibble: 336,776 x 19
    ## # Groups:   origin, tailnum [7,944]
    ##     year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
    ##    <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
    ##  1  2013     1     1      517            515         2      830            819
    ##  2  2013     1     1      533            529         4      850            830
    ##  3  2013     1     1      542            540         2      923            850
    ##  4  2013     1     1      544            545        -1     1004           1022
    ##  5  2013     1     1      554            600        -6      812            837
    ##  6  2013     1     1      554            558        -4      740            728
    ##  7  2013     1     1      555            600        -5      913            854
    ##  8  2013     1     1      557            600        -3      709            723
    ##  9  2013     1     1      557            600        -3      838            846
    ## 10  2013     1     1      558            600        -2      753            745
    ## # ... with 336,766 more rows, and 11 more variables: arr_delay <dbl>,
    ## #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>,
    ## #   air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

위 작업만 진행시 데이터프레임의 큰 변화가 없음(group만 표시해줌) 이 후
다른 작업 진행시 group을 인식한 상태로 작업 진행

### grouping and summarise

grouping 후 summarise 진행 시, group에 대해 summarise를 진행함

``` r
a=group_by(flights, origin)

summarise(a, delay=mean(dep_delay, na.rm=T))
```

    ## # A tibble: 3 x 2
    ##   origin delay
    ##   <chr>  <dbl>
    ## 1 EWR     15.1
    ## 2 JFK     12.1
    ## 3 LGA     10.3

    ## `summarise()` has grouped output by 'year', 'month'. You can override using the `.groups` argument.

    ## # A tibble: 365 x 4
    ## # Groups:   year, month [12]
    ##     year month   day flights
    ##    <int> <int> <int>   <int>
    ##  1  2013     1     1     842
    ##  2  2013     1     2     943
    ##  3  2013     1     3     914
    ##  4  2013     1     4     915
    ##  5  2013     1     5     720
    ##  6  2013     1     6     832
    ##  7  2013     1     7     933
    ##  8  2013     1     8     899
    ##  9  2013     1     9     902
    ## 10  2013     1    10     932
    ## # ... with 355 more rows

    ## `summarise()` has grouped output by 'year'. You can override using the `.groups` argument.

    ## # A tibble: 12 x 3
    ## # Groups:   year [1]
    ##     year month flights
    ##    <int> <int>   <int>
    ##  1  2013     1   27004
    ##  2  2013     2   24951
    ##  3  2013     3   28834
    ##  4  2013     4   28330
    ##  5  2013     5   28796
    ##  6  2013     6   28243
    ##  7  2013     7   29425
    ##  8  2013     8   29327
    ##  9  2013     9   27574
    ## 10  2013    10   28889
    ## 11  2013    11   27268
    ## 12  2013    12   28135

    ## # A tibble: 1 x 2
    ##    year flights
    ##   <int>   <int>
    ## 1  2013  336776

### summarise에 사용되는 여러 함수

-   sum, mean, sd, var, … : 기본적인 통계량 측정 함수(합, 평균,
    표준편차, 분산)
-   n() : count 함수
-   n\_distinct() : unique한 값 개수
-   first(x), last(x), nth(x, n) : x vector에서의 첫번째, 마지막, nth
    value

``` r
a=group_by(flights, tailnum)
summarise(a, dep_delay_mean=mean(dep_delay, na.rm=T), dep_delay_sd=sd(dep_delay, na.rm=T), count=n())
```

    ## # A tibble: 4,044 x 4
    ##    tailnum dep_delay_mean dep_delay_sd count
    ##    <chr>            <dbl>        <dbl> <int>
    ##  1 D942DN          31.5          30.9      4
    ##  2 N0EGMQ           8.49         33.3    371
    ##  3 N10156          17.8          36.3    153
    ##  4 N102UW           8            46.3     48
    ##  5 N103US          -3.20          4.29    46
    ##  6 N104UW           9.94         44.4     47
    ##  7 N10575          22.7          48.5    289
    ##  8 N105UW           2.58         22.2     45
    ##  9 N107US          -0.463        17.1     41
    ## 10 N108UW           4.22         25.7     60
    ## # ... with 4,034 more rows

### Using pipeline

-   pipeline (%&gt;%)을 이용하여 coding을 간결하게 진행할 수 있음
-   작업 1 %&gt;% 작업 2 - 작업 1의 결과에서 작업 2를 시작하게 됨
-   일일히 dataframe에 이름을 지정해주지 않고 바로 coding 가능해짐

``` r
# r group_summarise 1

a=group_by(flights, origin)

summarise(a, delay=mean(dep_delay, na.rm=T))
```

    ## # A tibble: 3 x 2
    ##   origin delay
    ##   <chr>  <dbl>
    ## 1 EWR     15.1
    ## 2 JFK     12.1
    ## 3 LGA     10.3

``` r
# using pipeline

flights %>% group_by(origin) %>% summarise(delay=mean(dep_delay, na.rm=T))
```

    ## # A tibble: 3 x 2
    ##   origin delay
    ##   <chr>  <dbl>
    ## 1 EWR     15.1
    ## 2 JFK     12.1
    ## 3 LGA     10.3

``` r
# 같은 결과가 나오는 것을 알 수 있음
```

``` r
# r group_summarise 2 example
daily=group_by(flights, year, month, day)

per_day = summarise(daily, flights=n())
```

    ## `summarise()` has grouped output by 'year', 'month'. You can override using the `.groups` argument.

``` r
per_day
```

    ## # A tibble: 365 x 4
    ## # Groups:   year, month [12]
    ##     year month   day flights
    ##    <int> <int> <int>   <int>
    ##  1  2013     1     1     842
    ##  2  2013     1     2     943
    ##  3  2013     1     3     914
    ##  4  2013     1     4     915
    ##  5  2013     1     5     720
    ##  6  2013     1     6     832
    ##  7  2013     1     7     933
    ##  8  2013     1     8     899
    ##  9  2013     1     9     902
    ## 10  2013     1    10     932
    ## # ... with 355 more rows

``` r
#가장 낮은 level인 day의 flight 계산

per_month = summarise(per_day, flights=sum(flights))
```

    ## `summarise()` has grouped output by 'year'. You can override using the `.groups` argument.

``` r
per_month
```

    ## # A tibble: 12 x 3
    ## # Groups:   year [1]
    ##     year month flights
    ##    <int> <int>   <int>
    ##  1  2013     1   27004
    ##  2  2013     2   24951
    ##  3  2013     3   28834
    ##  4  2013     4   28330
    ##  5  2013     5   28796
    ##  6  2013     6   28243
    ##  7  2013     7   29425
    ##  8  2013     8   29327
    ##  9  2013     9   27574
    ## 10  2013    10   28889
    ## 11  2013    11   27268
    ## 12  2013    12   28135

``` r
# 그 다음 level인 month의 flight summarise

per_year = summarise(per_month, flights=sum(flights))
per_year
```

    ## # A tibble: 1 x 2
    ##    year flights
    ##   <int>   <int>
    ## 1  2013  336776

``` r
# using pipeline

flights %>% group_by(year, month, day) %>% summarise(flights=n())
```

    ## `summarise()` has grouped output by 'year', 'month'. You can override using the `.groups` argument.

    ## # A tibble: 365 x 4
    ## # Groups:   year, month [12]
    ##     year month   day flights
    ##    <int> <int> <int>   <int>
    ##  1  2013     1     1     842
    ##  2  2013     1     2     943
    ##  3  2013     1     3     914
    ##  4  2013     1     4     915
    ##  5  2013     1     5     720
    ##  6  2013     1     6     832
    ##  7  2013     1     7     933
    ##  8  2013     1     8     899
    ##  9  2013     1     9     902
    ## 10  2013     1    10     932
    ## # ... with 355 more rows

``` r
flights %>% group_by(year, month, day) %>% summarise(flights=n()) %>% summarise(flights = sum(flights))
```

    ## `summarise()` has grouped output by 'year', 'month'. You can override using the `.groups` argument.

    ## `summarise()` has grouped output by 'year'. You can override using the `.groups` argument.

    ## # A tibble: 12 x 3
    ## # Groups:   year [1]
    ##     year month flights
    ##    <int> <int>   <int>
    ##  1  2013     1   27004
    ##  2  2013     2   24951
    ##  3  2013     3   28834
    ##  4  2013     4   28330
    ##  5  2013     5   28796
    ##  6  2013     6   28243
    ##  7  2013     7   29425
    ##  8  2013     8   29327
    ##  9  2013     9   27574
    ## 10  2013    10   28889
    ## 11  2013    11   27268
    ## 12  2013    12   28135

``` r
flights %>% group_by(year, month, day) %>% summarise(flights=n()) %>% summarise(flights = sum(flights)) %>%
  summarise(flights = sum(flights))
```

    ## `summarise()` has grouped output by 'year', 'month'. You can override using the `.groups` argument.
    ## `summarise()` has grouped output by 'year'. You can override using the `.groups` argument.

    ## # A tibble: 1 x 2
    ##    year flights
    ##   <int>   <int>
    ## 1  2013  336776

``` r
a1 = group_by(flights, year, month, day)
a2 = select(a1, arr_delay, dep_delay)
```

    ## Adding missing grouping variables: `year`, `month`, `day`

``` r
a3 = summarise(a2, 
               arr = mean(arr_delay, na.rm=T),
               dep = mean(dep_delay, na.rm=T))
```

    ## `summarise()` has grouped output by 'year', 'month'. You can override using the `.groups` argument.

``` r
a4 = filter(a3, arr > 30 | dep > 30)

print(a4)
```

    ## # A tibble: 49 x 5
    ## # Groups:   year, month [11]
    ##     year month   day   arr   dep
    ##    <int> <int> <int> <dbl> <dbl>
    ##  1  2013     1    16  34.2  24.6
    ##  2  2013     1    31  32.6  28.7
    ##  3  2013     2    11  36.3  39.1
    ##  4  2013     2    27  31.3  37.8
    ##  5  2013     3     8  85.9  83.5
    ##  6  2013     3    18  41.3  30.1
    ##  7  2013     4    10  38.4  33.0
    ##  8  2013     4    12  36.0  34.8
    ##  9  2013     4    18  36.0  34.9
    ## 10  2013     4    19  47.9  46.1
    ## # ... with 39 more rows

``` r
# using pipeline

flights %>% group_by(year, month, day) %>% select(arr_delay, dep_delay) %>%
  summarise(arr=mean(arr_delay, na.rm=T), dep=mean(dep_delay, na.rm=T)) %>%
  filter(arr > 30 | dep >30)
```

    ## Adding missing grouping variables: `year`, `month`, `day`

    ## `summarise()` has grouped output by 'year', 'month'. You can override using the `.groups` argument.

    ## # A tibble: 49 x 5
    ## # Groups:   year, month [11]
    ##     year month   day   arr   dep
    ##    <int> <int> <int> <dbl> <dbl>
    ##  1  2013     1    16  34.2  24.6
    ##  2  2013     1    31  32.6  28.7
    ##  3  2013     2    11  36.3  39.1
    ##  4  2013     2    27  31.3  37.8
    ##  5  2013     3     8  85.9  83.5
    ##  6  2013     3    18  41.3  30.1
    ##  7  2013     4    10  38.4  33.0
    ##  8  2013     4    12  36.0  34.8
    ##  9  2013     4    18  36.0  34.9
    ## 10  2013     4    19  47.9  46.1
    ## # ... with 39 more rows

### Other pipeline - using magrittr

-   %&lt;&gt;% : 최종 결과값을 처음에 지정한 dataframe에 저장할 것
-   %$% : dataframe의 특정 열만 사용하고 싶을 때 사용

``` r
library(magrittr)
```

    ## 
    ## Attaching package: 'magrittr'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     set_names

    ## The following object is masked from 'package:tidyr':
    ## 
    ##     extract

``` r
# %>% vs %<>%

x=1

x %<>% multiply_by(0.5) %>% print
```

    ## [1] 0.5

``` r
x %>% multiply_by(0.5) %>% print
```

    ## [1] 0.25

``` r
print(x)
```

    ## [1] 0.5

``` r
# %$%

flights %>% filter(dep_delay>0 & arr_delay>0) %$% cor(dep_delay, arr_delay) 
```

    ## [1] 0.9418739
