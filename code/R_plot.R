library(dplyr)
library(ggplot2)
library(reshape2)
library(plyr)
library(tidyverse)
library(ggpubr)
library(RColorBrewer)
data_dir = "C:\\Users\\nerissa.xu\\Documents\\Research Materials\\Project 1\\20240202\\plot\\"
output_dir = "C:\\Users\\nerissa.xu\\Documents\\Research Materials\\Project 1\\20240202\\output\\"

# convert the variable into proper expression for plot
stringConvert = function(p,lag){
  if(lag==0){
    if(p == 'Incidence_Rate'){
      return(bquote('Incidence Rate'))
    }  
    if(p == 'co'){
      return(bquote('CO Concentration (mg/m'^3*')'))
    }
    if (p == 'so2'){
      return(bquote('SO'[2]*' Concentration ('*mu*'g/m'^3*')'))
    }
    if(p == 'no2'){
      return(bquote('NO'[2]*' Concentration ('*mu*'g/m'^3*')'))
    }
    if(p == 'o3'){
      return(bquote('O'[3]*' Concentration ('*mu*'g/m'^3*')'))
    }
    if(p == 'pm10'){
      return(bquote('PM'[10]*' Concentration ('*mu*'g/m'^3*')'))
    }
    if(p == 'pm25'){
      return(bquote('PM'[2.5]*' Concentration ('*mu*'g/m'^3*')'))
    }
  }else{
    if(p == 'Incidence_Rate'){
        return(bquote(atop('Incidence Rate',
                               '['*.(lag)*'-Week Lag]')))
      }  
      if(p == 'co'){
        return(bquote(atop('CO Concentration (mg/m'^3*')',
                               '['*.(lag)*'-Week Lag]')))
      }
      if (p == 'so2'){
        return(bquote(atop('SO'[2]*' Concentration ('*mu*'g/m'^3*')',
                               '['*.(lag)*'-Week Lag]')))
      }
      if(p == 'no2'){
        return(bquote(atop('NO'[2]*' Concentration ('*mu*'g/m'^3*')',
                               '['*.(lag)*'-Week Lag]')))
      }
      if(p == 'o3'){
        return(bquote(atop('O'[3]*' Concentration ('*mu*'g/m'^3*')',
                               '['*.(lag)*'-Week Lag]')))
    }
    if(p == 'pm10'){
      return(bquote(atop('PM'[10]*' Concentration ('*mu*'g/m'^3*')',
                             '['*.(lag)*'-Week Lag]')))
      }
      if(p == 'pm25'){
        return(bquote(atop('PM'[2.5]*' Concentration ('*mu*'g/m'^3*')',
                           '['*.(lag)*'-Week Lag]')))
    }    
  }
}
titleConvert = function(p){
  if(p == 'Incidence_Rate'){
    return(bquote('Incidence Rate'))
  }  
  if(p == 'co'){
    return(bquote('CO'))
  }
  if (p == 'so2'){
    return(bquote('SO'[2]))
  }
  if(p == 'no2'){
    return(bquote('NO'[2]))
  }
  if(p == 'o3'){
    return(bquote('O'[3]))
  }
  if(p == 'pm10'){
    return(bquote('PM'[10]))
  }
  if(p == 'pm25'){
    return(bquote('PM'[2.5]))
  }
  
  
}
# plot partial dependence plot for IRR
gp_pdp = function(disease,lag,pollutant1){
  factorls = c('Incidence_Rate','co','no2','o3','pm10','pm25','so2')
  factorls = factorls[factorls!=pollutant1]
  pdp_dir = paste(data_dir,'GP/',disease,'/lag',lag,'/Incidence_Rate/pdp/',pollutant1,sep='')
  pdp = list()
  for(i in 0:lag){
    rowname = paste('lag',i,sep='')
    if(i==0){
      csv_dir = pdp_dir
    }
    else{
      csv_dir = paste(pdp_dir,'_lag',i, sep='')
    }
    files = list.files(csv_dir)  
    csvfiles = files[endsWith(files,'csv')]
    for(f in csvfiles){
      p2check = sapply(factorls,function(x) grepl(x,f))
      p2 = factorls[p2check]
      plotname = paste(pollutant1,p2,sep='_')
      df = read.csv(paste(csv_dir,f,sep ='/'))
      p = ggplot(df,aes(x=feature_val)) +
        geom_line(aes(y=X5pctl_irr,color='5th percentile'),size=1) +
        geom_ribbon(data=df,aes(ymin=X5pctl_ci_lower,ymax=X5pctl_ci_upper,fill='5th percentile'),alpha=0.15)+
        geom_line(aes(y=X50pctl_irr,color='50th percentile'),size=1) +
        geom_ribbon(data=df,aes(ymin=X50pctl_ci_lower,ymax=X50pctl_ci_upper,fill='50th percentile'),alpha=0.15)+
        geom_line(aes(y=X95pctl_irr,color='95th percentile'),size=1) +
        geom_ribbon(data=df,aes(ymin=X95pctl_ci_lower,ymax=X95ptcl_ci_upper,fill='95th percentile'),alpha=0.15)+
        labs(x=stringConvert(pollutant1,i),y='Incidence Rate Ratio',
             title=bquote(.(titleConvert(pollutant1))*' - '*.(titleConvert(p2)))) + 
        scale_color_manual(name=bquote(.(titleConvert(p2))),values=c("5th percentile"='#e41a1c','50th percentile'='#e6ab02','95th percentile'='#4daf4a'),
                           limits=c('5th percentile', '50th percentile', '95th percentile')) +
        scale_fill_manual(name=bquote(.(titleConvert(p2))),values=c("5th percentile"='#e41a1c','50th percentile'='#e6ab02','95th percentile'='#4daf4a'),
                          limits=c('5th percentile', '50th percentile', '95th percentile')) +
        theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                           panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
        theme(legend.position='right')
      
      print(p)
      
    }

  }
  
}
