# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:19:56 2025

@author: user
"""

# =============================================================================
# 
import os
import netCDF4 as nc
import pandas as pd
import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.style as mplstyle
import cartopy.crs as ccrs
import netCDF4 as nc
from netCDF4 import Dataset
import shapefile
import time
# =============================================================================
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告


import shapefile
# from wrf import getvar, ALL_TIMES
from datetime import datetime, timedelta
import cmaps
import sys
# 将目标模块所在的路径添加到 sys.path 中
sys.path.append(r"/home/yfdong/data/work/LF-SAM/code/Library")
from MeteoVarPlot import draw_2DVAR, draw_Bias, draw_VARtemporal, draw_Bias_subplot,draw_2DVAR_subplot ,draw_2DuvVAR ,draw_uvBias_subplot ,draw_2DuvVAR_subplot
from MeteoChartPlot import add_TickGrid
from PreprocessVar import (
    get_MONTH_abbr, get_pentad, get_letter,
    preprocess_var, read_var,Events_configs,
    Read_VarsFromVarsDict, experiments_types,return_Event_Date,
    load_geodata, SubsetDomainConfigure ,create_domain_mask,get_letter,
    get_CoordPair
)
def get_domain_region(domain_name, config_list):
    for item in config_list:
        if item["DomainName"] == domain_name:
            return item["DomainRegion"]
    raise ValueError(f"DomainName '{domain_name}' not found.")

def extract_SubsetRegion_mask(var ,domain_mask):
    row_indices, col_indices = np.where(domain_mask == 1)  # 提取 ==1 的区域
    row_start, row_end = row_indices.min(), row_indices.max()
    col_start, col_end = col_indices.min(), col_indices.max()
    # 步骤2：切片获取子区域数据（保持二维结构）
    sub_var = var[row_start:row_end+1, col_start:col_end+1]
    return sub_var
# --------------------------------- 超参数 ----------------------------------
# 选择输出文件时间间隔
Timedelta=1
# 读取变量列表
LSMvarList = ['SH2O_Layer1', 'SH2O_Layer2','SH2O_Layer3','SH2O_Layer4','SH2O_MEAN',
              ''
              'RUNSF', 'RUNSB',
              'Q2', 'rh2', 'td2','ET', 'RAINNC', 
              'T2',"TSLB_Layer1", "FIRA", "FSA", "LH", "SH" ]
ATMvarList = ["QVAPOR_850hPa","QVAPOR_500hPa","rh_850hPa","rh_500hPa",
              "tk_850hPa", "tk_500hPa",
              'ua_850hPa' ,'ua_700hPa', 'ua_500hPa',
              'va_850hPa', 'va_700hPa', 'va_500hPa',
               "height_850hPa","height_700hPa","height_500hPa",
               'cloudfrac_Low',  'LCL', 'CAPE', 'PBLH', 
            "DivgQ"]
ReadVarList = LSMvarList+ATMvarList

def draw_subplot(NumRows, NumCols, results, PlotVarList, SaveName, IF_t_significance_test=True):
    # fig, axes = plt.subplots(nrows=NumRows, ncols=NumCols, figsize=(9.28/NumRows, 8/NumCols), dpi=350 ,sharex=True ,subplot_kw={'projection': ccrs.PlateCarree()})
    # 9.28,2
    fig, axes = plt.subplots(nrows=NumRows, ncols=NumCols, 
                             figsize=(9.3/4*NumCols/1.25, 2*NumRows/1.3), 
                             dpi=500 ,gridspec_kw={'wspace': 0.1, 'hspace': 0.15},
                             subplot_kw={'projection': ccrs.PlateCarree()})#,sharex=True ,subplot_kw={'projection': ccrs.PlateCarree()})
    # print(NumRows, NumCols)
    # axs = axes.flatten()
    for IDx, ExpermentState  in enumerate(['Ctrl', 'Expl', 'Diff']):
        # print(ExpermentState)
        # if yearIndex==0:
        if ExpermentState=='Diff':
            Ylaybel = 'Diff'
        elif ExpermentState=='Expl':
            Ylaybel = 'WRF-H'
        elif ExpermentState=='Ctrl':
            Ylaybel = 'WRF-S' 
            
        if ExpermentState=='Diff':
            IFdiff = True
        else:
            IFdiff = False

        for IDy in range(len(PlotVarList)):
            axe= axes[IDx, IDy]
            pos = axe.get_position()

            
            # print(IDx, IDy, pos, [pos.x0, pos.y0 -0.035, pos.width, 0.008])
            # if  IDy==0:
            #     axes[IDx, IDy].set_ylabel(Ylaybel, fontsize=Plt_CommonParams['xy_labelsize']+1, rotation=90, labelpad=7) #,fontweight='bold'

            if IDx <2:
                pad = 0.025
            else:
                pad = 0.045
            IDindex = IDx*len(PlotVarList)+IDy
            VAR_NAME = PlotVarList[IDy]["VAR_NAME"]
            UNIT = PlotVarList[IDy]["UNIT"]
            subTitle = PlotVarList[IDy]["Title"]
            SCALE_FACTOR = PlotVarList[IDy]["SCALE_FACTOR"]
            plotMAX2 = PlotVarList[IDy]["plotMAX2"]
            # ------------------------------ 添加子图标号 ------------------------------
            axes[IDx,IDy].text(0.03, 0.95, f"({get_letter(IDindex+0)}) {Ylaybel}", 
                            fontsize=7, fontweight='bold',  # 加粗字体
                            transform=axes[IDx,IDy].transAxes, va='top',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            subTitle = None  
            # print(subLON[:].shape, TtestVar.shape)
            # if VAR_NAME =='td2' and IFdiff == False:
            #     # PlotVar = PlotVar +273
            #     print(np.nanmax(PlotVar), np.nanmin(PlotVar))
            # if VAR_NAME =='ET' and IFdiff == False:
            #     # PlotVar = PlotVar +273
            #     print(np.nanmax(PlotVar), np.nanmin(PlotVar))               
            if VAR_NAME =='SM':
                SM1 = extract_SubsetRegion_mask(results[ExpermentState]['SH2O_Layer1'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) # 0-10cm
                SM2 = extract_SubsetRegion_mask(results[ExpermentState]['SH2O_Layer2'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) # 10-40cm
                SM3 = extract_SubsetRegion_mask(results[ExpermentState]['SH2O_Layer3'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) # 40-100cm
                SM4 = extract_SubsetRegion_mask(results[ExpermentState]['SH2O_Layer4'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) # 100-200cm
                SM = (SM1*10 +SM2*30 +SM3*60 +SM4*100)/200
                PlotVar = SM
            elif VAR_NAME =='td2' and IFdiff == False:
                PlotVar = extract_SubsetRegion_mask(results[ExpermentState][VAR_NAME][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) +273
            else:
                PlotVar = extract_SubsetRegion_mask(results[ExpermentState][VAR_NAME][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) 
                # TtestVar = extract_SubsetRegion_mask(results['Ttest'][VAR_NAME][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) 

            if VAR_NAME in ["height_850hPa","height_700hPa","height_500hPa", 
                            'tk_850hPa',"tk_500hPa", 
                            "QVAPOR_850hPa",  "QVAPOR_500hPa",
                            'rh_850hPa','rh_500hPa']:
                if VAR_NAME =="PSFC":
                    PlotVarUwind = extract_SubsetRegion_mask(results[ExpermentState]['U10'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) 
                    PlotVarVwind = extract_SubsetRegion_mask(results[ExpermentState]['V10'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) 
                else:
                    HeightName =VAR_NAME.split('_')[1]
                    PlotVarUwind = extract_SubsetRegion_mask(results[ExpermentState][f'ua_{HeightName}'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample]) 
                    PlotVarVwind = extract_SubsetRegion_mask(results[ExpermentState][f'va_{HeightName}'][:]*SCALE_FACTOR, DomainMask[ ::subsample, ::subsample])                     

            if VAR_NAME in ["height_850hPa","height_700hPa","height_500hPa","tk_850hPa","tk_500hPa"]:
                std = ((np.nanstd(results['Ctrl'][VAR_NAME][:]))* SCALE_FACTOR)
                mean = ((np.nanmean(results['Ctrl'][VAR_NAME][:]))* SCALE_FACTOR)
                plotMAX1 = mean+std*1.5
                plotMIN = mean-std*1.5
            else:
                plotMAX1 = PlotVarList[IDy]["plotMAX1"]
                plotMIN = PlotVarList[IDy]["plotMIN"]
 
            # axes[IDx, IDy].text(0.025, 0.95, f"({get_letter(IDindex+0)})", fontsize=8, 
            #         transform=axes[IDx, IDy].transAxes, va='top',
            #         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            CMAPmean=PlotVarList[IDy]["CMAPmean"]
            # CMAPdiff=PlotVarList[IDy]["CMAPdiff"]
            # CMAPdiff=cmaps.amwg256
            CMAPdiff = cmaps.NCV_manga
            # if IDx==0:
            #     subTitle =f"{subTitle} ({UNIT})"
            # else:
            #     subTitle =None
            # if 
            if VAR_NAME in ["height_850hPa","height_700hPa","height_500hPa", 
                            'tk_850hPa',"tk_500hPa", 
                            "QVAPOR_850hPa",  "QVAPOR_500hPa",
                            'rh_850hPa','rh_500hPa']:
                 # 绘图
                if IFdiff:
                    uvMAX =uv_MaxDiff
                    RegridShape = RegridShapeDiff
                    Qscale =QscaleDiff
                    CMAP = CMAPdiff
                    draw_uvBias_subplot(DomainRegion,plotMAX2 ,uvMAX, RegridShape,
                    subLON[:], subLAT[:], 
                    PlotVarUwind[:], PlotVarVwind[:] , PlotVar[:] , subDEM[:],
                        subTitle, cbarTitle, quiverTitle ,CMAP ,
                    **Plt_CommonParams , pad=pad,ax=axe, Cbar=cbar,
                    Topo=False, IF_Rectangle=False ,Qscale =Qscale
                    )
                    
                else:
                    # print(IFdiff)
                    uvMAX = uv_MaxMean
                    RegridShape = RegridShapeMean
                    Qscale = QscaleMean
                    CMAP = CMAPmean
                    draw_2DuvVAR_subplot(DomainRegion, plotMIN, plotMAX1 ,uvMAX, RegridShape,
                    subLON[:], subLAT[:], 
                    PlotVarUwind[:], PlotVarVwind[:] , PlotVar[:] , subDEM[:],
                        subTitle, cbarTitle, quiverTitle ,CMAP ,
                    **Plt_CommonParams , pad=pad,ax=axe, Cbar=cbar,
                    Topo=False, IF_Rectangle=False ,Qscale =Qscale
                    )
                    def custom_fmt(x):
                        return r'$\bf{%.0f}$' % x   
                    if VAR_NAME in ['height_500hPa']:
                        ax = axes[IDx, IDy]
                        contour = ax.contour(subLON[:], subLAT[:], PlotVar[:], levels = [5880] ,alpha = 1 , colors = ["#FF7F24"], linewidths=1, antialiased=True)
                        # ax.clabel(contour, inline=True, fontsize=ContourLabelsize, colors='#8B2323', fmt='%1.0f',manual=False)
                        # 添加等高线标签，并设置背景透明度
                        cl = ax.clabel(contour, inline=True, fontsize=ContourLabelsize, colors='#8B2323', fmt=custom_fmt, manual=False)
                        for text in cl:
                            text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none',pad= 1))
                            # 手动调整 text 的位置（假设需要偏移）
                            x, y = text.get_position()  # 获取当前 text 的位置
                            text.set_position((x - 1, y - 0.1))  # 调整位置 (可根据需求调整偏移量)

                    elif VAR_NAME in ['height_850hPa']:
                        ax = axes[IDx, IDy]
                        CtrlTotalWind = np.sqrt(PlotVarUwind[:]**2 +PlotVarVwind[:]**2)

                        contour = ax.contour(subLON[:], subLAT[:], PlotVar[:], levels = [1450] ,alpha = 1 , colors = ["#FF7F24"], linewidths=1, antialiased=True)
                        # 添加等高线标签，并设置背景透明度
                        cl = ax.clabel(contour, inline=True, fontsize=ContourLabelsize, colors='#8B2323', fmt=custom_fmt, manual=False)
                        for text in cl:
                            text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none',pad= 1))
                            # 手动调整 text 的位置（假设需要偏移）
                            x, y = text.get_position()  # 获取当前 text 的位置
                            text.set_position((x + 1, y - 0.1))  # 调整位置 (可根据需求调整偏移量)
                            
            else:
                if IFdiff:
                    CMAP = CMAPdiff
                    draw_Bias_subplot(
                        DomainRegion, plotMAX2, subLON[:], subLAT[:], 
                        PlotVar[:] , subTitle, cbarTitle=f'[{UNIT}]', 
                        CMAP=CMAP, save_name=False,
                        **Plt_CommonParams, pad=pad, DEM = subDEM[:], ax=axe, 
                        Topo=True, Cbar=cbar, IF_Rectangle=False
                        )   
                else:
                    CMAP = CMAPmean
                    draw_2DVAR_subplot(
                        DomainRegion, plotMIN ,plotMAX1, subLON[:], subLAT[:], 
                        PlotVar[:] , subTitle, cbarTitle=f'[{UNIT}]', 
                        CMAP=CMAP, save_name=False,
                        **Plt_CommonParams, pad=pad, DEM = subDEM[:], ax=axe, 
                        Topo=True, Cbar=cbar, IF_Rectangle=False
                        )
            # if IFdiff and IF_t_significance_test:
            #     TtestVar =TtestVar
            #     # print(VAR_NAME)
            #     # print(TtestVar)
            #     # axe.scatter(subLON[TtestVar], 
            #     #         subLAT[TtestVar],
            #     #         s=0.02, c='black', alpha=1, 
            #     #         marker='.',     
            #     #         transform=ccrs.PlateCarree(),
            #     #         label=f'Significant (p<{0.05})')
            #     # axe.contour(subLON, subLAT,
            #     #             TtestVar,
            #     #             levels=[0.95, 1.05],
            #     #             # colors='#00688B',  # 无轮廓线
            #     #             # alpha=0.3,  # 30%透明度
            #     #             linewidths=0.5,
            #     #             # edgecolor='#00688B',
            #     #             # linestyles='dashed',
            #     #             )
            #     # 添加显著性区域 (95%置信水平)
            #     # 使用官方推荐的"Statistical Significance"样式
            #     # sig_contour = axe.contour(subLON, subLAT, TtestVar,
            #     #                         levels=[1],  # 代表p<0.05
            #     #                         colors='none')  # 无轮廓线

            #     # # 填充显著性区域 - 遵循Nature/Science配色标准
            #     # # plt.gcf().colorbar(cf, ax=axe)  # 添加基础场色标
            #     axe.contourf(subLON, subLAT, TtestVar,
            #                 levels=[0.99, 1.01],  # 精确界定显著性边界
            #                 hatches=['..'],       # 点状图案表示显著性
            #                 colors='None',        # 无填充颜色
            #                 alpha=0,              # 完全透明填充
            #                 edgecolor='grey',    # 符号颜色
            #                 linewidths=0.5,
            #                 zorder=9)       # 细线
# 绘制散点图
            if IDx <(NumRows-1):
                # axe.set_xticks([])
                # axe.set_xticklabels([])  # 同步清除标签
                axe.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                # print("当前x轴刻度：", axe.get_xticks())   # 应该输出空数组
                if IDy != 0:
                    axe.set_yticks([])
            elif IDx ==(NumRows-1):
                if IDy != 0:
                    axe.set_yticks([])      
                # 添加SAM区域边框
                # print("添加SAM区域边框")
                SAM_Region = get_domain_region("SAM", SubsetDomainConfigure)
                def add_SubsetDomain(ax, SubsetDomain, linewidth, edgecolor, linestyle):
                    import matplotlib.patches as patches
                    rect = patches.Rectangle((SubsetDomain[2], SubsetDomain[0]),  # 左下角的坐标 (经度, 纬度)
                                            SubsetDomain[3] - SubsetDomain[2],  # 宽度 (经度的范围)
                                            SubsetDomain[1] - SubsetDomain[0],  # 高度 (纬度的范围)
                                            linewidth=linewidth,  # 边框宽度
                                            edgecolor=edgecolor,  # 边框颜色
                                            facecolor='none',  # 矩形内部颜色，设置为 'none' 即可透明
                                            linestyle=linestyle)  # 边框线型

                    # 将矩形添加到坐标轴中
                    ax.add_patch(rect) 
                    pass   
                add_SubsetDomain(axe, SAM_Region, linewidth=1.5, edgecolor='black',linestyle='--')    
                JTB_Region = get_domain_region("JTB", SubsetDomainConfigure)
                add_SubsetDomain(axe, JTB_Region, linewidth=1.5, edgecolor='black',linestyle='--')
                # 给colorbar添加标签
                PlotSubTitle = PlotVarList[IDy]["Title"]
                axe.text(0.5, -0.38, f"{PlotSubTitle} ({UNIT})", ha='center', va='bottom', transform=axe.transAxes, fontsize=6, fontweight='bold') # [left, bottom, width, height]
        # elif IDy == 0:
        #     ax = axes[IDx, IDy]
        #     ax.set_xticks(np.arange(DomainRegion[0], DomainRegion[1]+1, 1))
        
    plt.subplots_adjust(wspace=0.05, hspace=0.15) #, hspace=0.15
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,wspace=0.25, hspace=0.1)
    # os.makedirs(os.path.join(FigurePath, FeedbackState), exist_ok=True)
    # plt.savefig(os.path.join(FigurePath, FeedbackState, f'{SaveName}.jpg'))
    plt.show()
    plt.close()
    pass
# =============================================================================
# custom Functions:
import matplotlib.dates as mdates
BasePath = "/home/yfdong/data/work/LF-SAM/code/Library"
DataPath = '/raid61/yfdong/data/work/LF-SAM/output/'
FigurePath = '/home/yfdong/data/work/LF-SAM/code/AnalysisModule/Figure/SyntheticAnalysis'
# 读取地形数据
LAT, LON, LandMask, DEM = load_geodata(GeoPath ="/raid61/yfdong/data/work/LF-SAM/Domain/def/geo_em.nc" ,IF_LandMask=True)
# 读取区域设置
DomainRegion2 = [25, 35, 109, 122] # 提取mask范围
DomainMask = create_domain_mask(LAT, LON, DomainRegion2)
CalibrationStatus = "def" # "def" ,"calib"、
experiment = experiments_types[0]
LAYER =1
subsample = 2
subLAT =extract_SubsetRegion_mask(LAT[ ::subsample, ::subsample] ,DomainMask[ ::subsample, ::subsample])
subLON =extract_SubsetRegion_mask(LON[ ::subsample, ::subsample] ,DomainMask[ ::subsample, ::subsample])
subDEM =extract_SubsetRegion_mask(DEM[ ::subsample, ::subsample] ,DomainMask[ ::subsample, ::subsample])
# =============================================================================
EventState = 'Dur'
EventState = 'Pre'
FeedbackState = 'Negative'
# FeedbackState = 'Positive'
print('=============================================================================')
print(f'Event State: {EventState}')
print(f'Feedback State: {FeedbackState}')
print('=============================================================================')
# =============================================================================
import pandas as pd
Event_df = pd.read_excel(f'/home/yfdong/data/work/LF-SAM/code/AnalysisModule/HuangshanPrecAnalysis/EventDateDf_{FeedbackState}.xlsx')
Event_subset_df = Event_df.copy()
EventNum = len(Event_subset_df)


# 生成唯一的缓存文件名（包含关键参数）
cache_filename = (
    f"WRF_cache_SyntheticAnalysis_{FeedbackState}_{EventState}_{CalibrationStatus}_"
    f"{Timedelta}h_{EventNum}events_mean.nc"
)
cache_path = os.path.join(DataPath, 'Event/HeavyRainfall', 'cache', cache_filename)
# 尝试读取缓存文件
start = time.time()
if os.path.exists(cache_path):
    print(f"读取缓存文件: {cache_path}")
    with nc.Dataset(cache_path, 'r') as ds:
        # 读取结果数据
        results = {
            'Ctrl': {var: ds[f'Ctrl_{var}'][:] for var in ReadVarList},
            'Expl': {var: ds[f'Expl_{var}'][:] for var in ReadVarList},
        }
else:
    print("未找到缓存文件，重新处理数据...")
    results = {'Ctrl': {}, 'Expl': {}}
    # ======================== 数据计算部分 ========================
    for var in ReadVarList:
        # 初始化列表存储每个事件的原始数据（不进行时间平均）
        WRF_H_templateVar = np.zeros((EventNum, LAT.shape[0], LAT.shape[1]))
        WRF_S_templateVar = np.zeros((EventNum, LAT.shape[0], LAT.shape[1]))

        for EventIndex in range(EventNum):
            # 获取事件时间范围（保持不变）
            if EventState == 'Pre':
                EventName, Read_StartDate, Read_EndDate = (
                    Event_subset_df.loc[EventIndex, 'EventName'],
                    Event_subset_df.loc[EventIndex, 'PreStartDate'],
                    Event_subset_df.loc[EventIndex, 'DurStartDate']
                )
            elif EventState == 'Dur':
                EventName, Read_StartDate, Read_EndDate = (
                    Event_subset_df.loc[EventIndex, 'EventName'],
                    Event_subset_df.loc[EventIndex, 'DurStartDate'],
                    Event_subset_df.loc[EventIndex, 'DurEndDate']
                )
            else:
                raise ValueError('EventState 必须是 "Pre" 或 "Dur"')
            
            DateList = pd.date_range(Read_StartDate, Read_EndDate, freq=f'{Timedelta}h')
            time_steps = len(DateList)  # 实际时间步数
            
            ReadVar_CommonParams = {
                'ReadStartDate': Read_StartDate,
                'ReadEndDate': Read_EndDate,
                'DateList': DateList,
                'DATA_PATH': os.path.join(DataPath, 'Event/HeavyRainfall', FeedbackState, EventName),
                'CalibrationStatus': CalibrationStatus,
                'LAYER': LAYER,
                'SCALE_FACTOR': True,
                'LANDMASK': LandMask,
            }
            
            ReadCtrlVarParms = {**ReadVar_CommonParams, 'Experimental_Method': experiment['control_method'], 'VAR_NAME': var}
            ReadExplVarParms = {**ReadVar_CommonParams, 'Experimental_Method': experiment['experiment_method'], 'VAR_NAME': var}

            # 每次事件降水前期时间平均
            WRF_H_templateVar[EventIndex, :]  = np.nanmean((preprocess_var(**ReadCtrlVarParms)[0]), axis = 0)
            WRF_S_templateVar[EventIndex, :]  = np.nanmean((preprocess_var(**ReadExplVarParms)[0]), axis = 0)

        # 计算每个变量的事件集平均值
        results['Ctrl'][var] = WRF_H_templateVar[ :, ::subsample, ::subsample]
        results['Expl'][var] = WRF_S_templateVar[ :, ::subsample, ::subsample]
        
    # ===================== 保存结果到缓存文件 =====================
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with nc.Dataset(cache_path, 'w') as ds:
        # 创建维度
        ds.createDimension('lat', LAT[ ::subsample, ::subsample].shape[0])
        ds.createDimension('lon', LAT[ ::subsample, ::subsample].shape[1])
        ds.createDimension('Date', results['Expl'][var].shape[0])
        # 保存坐标变量
        lat = ds.createVariable('lat', 'f4', ('lat', 'lon'))
        lon = ds.createVariable('lon', 'f4', ('lat', 'lon'))
        Date = ds.createVariable('Date', 'f4', ('Date'))
        lat[:] = LAT[ ::subsample, ::subsample]
        lon[:] = LON[ ::subsample, ::subsample]
        Date[:] = np.arange(0, results['Expl'][var].shape[0])
        # 保存计算结果
        for var in ReadVarList:
            # print(var)
            ctrl_var = ds.createVariable(f'Ctrl_{var}', 'f4', ('Date', 'lat', 'lon'), zlib =True)
            expl_var = ds.createVariable(f'Expl_{var}', 'f4', ('Date', 'lat', 'lon'), zlib =True)
            ctrl_var[:] = results['Ctrl'][var]
            expl_var[:] = results['Expl'][var]
        
        # 添加文件属性便于追溯
        ds.setncattr('FeedbackState', FeedbackState)
        ds.setncattr('EventState', EventState)
        ds.setncattr('CalibrationStatus', CalibrationStatus)
        ds.setncattr('Timedelta', str(Timedelta))
        ds.setncattr('EventNum', str(EventNum))
        ds.setncattr('Generated_Time', time.ctime())

    print(f"数据已缓存至: {cache_path}")

# ------------------------- 计算差异（无论是否缓存都需要) -----------------------------------
results['Diff'] = {
    var: results['Ctrl'][var] - results['Expl'][var]
    for var in ReadVarList
}

end = time.time()
print(f"总耗时: {end - start:.2f} 秒")

def calculate_t_significance_test(X1, X2, alpha=0.05):
    Diff_x1_x2 = X1-X2
    # 计算有效样本数
    valid_samples = np.sum((Diff_x1_x2), axis=0)
    # 计算差异的均值和标准差
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_diff = np.nanmean(Diff_x1_x2, axis=0)
        std_diff = np.nanstd(Diff_x1_x2, axis=0, ddof=1)
    # 计算t统计量
    t_stat = np.zeros((X1.shape[1], X1.shape[2]))
    mask = valid_samples > 1
    t_stat[mask] = mean_diff[mask] / (std_diff[mask] / np.sqrt(valid_samples[mask]))
    # 使用t分布计算p值
    from scipy.stats import t
    p_values = np.ones((X1.shape[1], X1.shape[2]))
    p_values[mask] = 2 * (1 - t.cdf(np.abs(t_stat[mask]), df=valid_samples[mask]-1))
    # 创建显著性标记数组
    significant = (p_values < alpha) & mask
    # significant = (significant == 1)
    return significant
# 定义结果字典
results_mean = {
    'Ctrl': {},
    'Expl': {},
    'Diff': {},
    'Ttest':{},
}
for var in ReadVarList:
    # print(var)

    if var in LSMvarList:
        results_mean['Ctrl'][var] = np.nanmean( results['Ctrl'][var] , axis=0) *LandMask[ ::subsample, ::subsample]
        results_mean['Expl'][var] = np.nanmean( results['Expl'][var] , axis=0) *LandMask[ ::subsample, ::subsample]
    else:
        results_mean['Ctrl'][var] = np.nanmean( results['Ctrl'][var] , axis=0) 
        results_mean['Expl'][var] = np.nanmean( results['Expl'][var] , axis=0) 
        
    results_mean['Ttest'][var] = calculate_t_significance_test(results['Expl'][var], results['Ctrl'][var], alpha=0.05)    
    results_mean['Diff'][var] = np.nanmean((results['Expl'][var] - results['Ctrl'][var]) , axis=0) 
    # 计算统计显著性检验

import cmaps
# 定义公共参数
Plt_CommonParams = {
    'width': 3.2,
    'Height': 2.3,
    'dpi': 250,
    'title_labelsize': 7,
    'xy_labelsize': 4.,
    'cbar_Legend_size': 4.5,
    'Grid': True,
    'Save': False,
    'Shp': False,
    'River': False,
    'Basin': False,
    # 'Topo': True
}

# 定义变量参数
cbarTitle = True
cbar = True
subsample = 2 # 可视化提取数据间隔

quiverTitle = 'm/s'
CMAPmean=cmaps.MPL_StepSeq
# CMAPdiff=cmaps.amwg256
CMAPdiff = cmaps.NCV_manga
suptitle = None
ContourLabelsize = 5
DomainRegion = [25, 35, 109, 122]

# 修改部分1：在开始绘图前设置全局rcParam参数
plt.rcParams['axes.edgecolor'] = 'black'  # 边框颜色
plt.rcParams['axes.linewidth'] = 0.25     # 边框线宽

# ----------------------- 地表水分循环分析 -----------------------
DiffPlotVarList = [
    {"VAR_NAME": "RAINNC", "UNIT": "mm/hour", "plotMAX1": 2.5, "plotMAX2": 0.5, "plotMIN": 0, "SCALE_FACTOR": 1, "CMAPmean":cmaps.WhiteBlueGreenYellowRed,"CMAPdiff":cmaps.amwg256, "Title" :"Prec"},
    {"VAR_NAME": "SM", "UNIT": "m³/m³", "plotMAX1": 0.45, "plotMAX2": 0.03, "plotMIN": 0.2, "SCALE_FACTOR": 1, "CMAPmean":cmaps.MPL_StepSeq,"CMAPdiff":cmaps.CBR_drywet,"Title" :"SM"},
    {"VAR_NAME": "ET", "UNIT": "mm/hour", "plotMAX1": 0.25, "plotMAX2": 0.02, "plotMIN": 0.00, "SCALE_FACTOR": 1/24, "CMAPmean":cmaps.MPL_StepSeq,"CMAPdiff":cmaps.CBR_drywet, "Title" :"ET"},
    {"VAR_NAME": "Q2", "UNIT": "g/kg", "plotMAX1": 20, "plotMAX2": 0.3, "plotMIN": 10, "SCALE_FACTOR": 1, "CMAPmean":cmaps.MPL_StepSeq,"CMAPdiff":cmaps.CBR_drywet, "Title" :"Q2m"},
]

# 设置子图行数和列数
NumRows = 3
NumCols = 4
PlotVarList = DiffPlotVarList
SaveName = f'WaterBlance_{EventState}'
draw_subplot(NumRows, NumCols, results_mean, PlotVarList, SaveName)

# ----------------------- 热力循环分析 --------------------------
DiffPlotVarList = [
    {"VAR_NAME": "TSLB_Layer1", "UNIT": "K", "plotMAX1": 305, "plotMAX2": 1., "plotMIN": 295, "SCALE_FACTOR": 1, "CMAPmean":cmaps.CBR_coldhot, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"ST at 0-10cm"},     
    {"VAR_NAME": "FIRA", "UNIT": "W/m²", "plotMAX1": 100, "plotMAX2": 5, "plotMIN": 0, "SCALE_FACTOR": 1, "CMAPmean":cmaps.CBR_coldhot, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"NLW"}, #"Net longwave radiation flux"
    {"VAR_NAME": "FSA", "UNIT": "W/m²", "plotMAX1": 300, "plotMAX2": 10, "plotMIN": 100, "SCALE_FACTOR": 1, "CMAPmean":cmaps.CBR_coldhot, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"NSW"}, #"Net shortwave radiation flux"
    {"VAR_NAME": "LH", "UNIT": "W/m²", "plotMAX1": 150, "plotMAX2": 20, "plotMIN": 0.00, "SCALE_FACTOR": 1, "CMAPmean":cmaps.CBR_coldhot, "CMAPdiff":cmaps.CBR_coldhot,  "Title" :"LH"},#"Latent heat"
    {"VAR_NAME": "SH", "UNIT": "W/m²", "plotMAX1": 100, "plotMAX2": 10, "plotMIN": 0.00, "SCALE_FACTOR": 1, "CMAPmean":cmaps.CBR_coldhot, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"SH"},#"Sensible heat"
    ]
# 设置子图行数和列数
NumRows = 3
NumCols = 5
SaveName = f"EnergyBlance_{EventState}"
PlotVarList = DiffPlotVarList
TtestVar = draw_subplot(NumRows, NumCols, results_mean, PlotVarList, SaveName)

# ------------------- 边界层气象变量分析 --------------------------
DiffPlotVarList = [
    {"VAR_NAME": "cloudfrac_Low", "UNIT": "%", "plotMAX1": 100, "plotMAX2": 5, "plotMIN": 0, "SCALE_FACTOR": 100, "CMAPmean":cmaps.cmocean_gray_r, "CMAPdiff":cmaps.MPL_RdGy_r, "Title" :"LCF"},
    {"VAR_NAME": "LCL", "UNIT": "m", "plotMAX1": 1500, "plotMAX2": 100, "plotMIN": 300, "SCALE_FACTOR": 1, "CMAPmean":cmaps.cmocean_dense_r, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"LCL"},
    {"VAR_NAME": "CAPE", "UNIT": "J/kg", "plotMAX1": 1200, "plotMAX2": 100, "plotMIN": 200, "SCALE_FACTOR": 1, "CMAPmean":cmaps.cmocean_thermal, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"MCAPE"},
    {"VAR_NAME": "PBLH", "UNIT": "m", "plotMAX1": 1000, "plotMAX2": 50, "plotMIN": 300, "SCALE_FACTOR": 1, "CMAPmean":cmaps.nice_gfdl, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"PBLH"}, 
    ]
# 设置子图行数和列数
NumRows = 3
NumCols = 4
SaveName = f'ThermodynamicVariable_{EventState}'
PlotVarList = DiffPlotVarList
TtestVar = draw_subplot(NumRows, NumCols, results_mean, PlotVarList, SaveName)

# -------------------------- 大气环流分析 ----------------------------
DiffPlotVarList = [
    {"VAR_NAME": "height_850hPa", "UNIT": "gpm", "plotMAX1": 1425+50, "plotMAX2": 1, "plotMIN": 1425, "SCALE_FACTOR": 1, "CMAPmean":cmaps.MPL_viridis, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"GH at 850hPa & wind"},
    {"VAR_NAME": "height_700hPa", "UNIT": "gpm", "plotMAX1": 3075+50, "plotMAX2": 1, "plotMIN": 3075, "SCALE_FACTOR": 1, "CMAPmean":cmaps.MPL_viridis, "CMAPdiff":cmaps.MPL_BrBG, "Title" :"GH at 700hPa & wind"},    
    {"VAR_NAME": "height_500hPa", "UNIT": "gpm",  "plotMAX1": 5860+50, "plotMAX2": 1, "plotMIN": 5860, "SCALE_FACTOR": 1, "CMAPmean":cmaps.MPL_viridis, "CMAPdiff":cmaps.CBR_coldhot, "Title" :"GH at 500hPa & wind"},
    {"VAR_NAME": "DivgQ", "UNIT": "$10^{-5}$ kg·(Pa·m$^2$·s)$^{-1}$", "plotMAX1": 1., "plotMAX2": 0.3, "plotMIN": -0.1, "SCALE_FACTOR": -1.0, "CMAPmean":cmaps.WhiteBlueGreenYellowRed, "CMAPdiff":cmaps.WhiteBlueGreenYellowRed,  "Title" :"DivgQ"},
                    ]

uv_MaxMean =10
uv_MaxDiff = 1/2
QscaleMean = 80 # 越大，箭头越短 #diff:Qscale = 5
RegridShapeMean = 12 #越大，矢量越密集 
QscaleDiff = 6 # 越大，箭头越短 #diff:Qscale = 5
RegridShapeDiff = 15 #越大，矢量越密集

# 设置子图行数和列数
NumRows = 3
NumCols = 4
SaveName = f'AtmosphericCirculation_{EventState}'
PlotVarList = DiffPlotVarList
TtestVar = draw_subplot(NumRows, NumCols, results_mean, PlotVarList, SaveName)

# -------------------------- 土壤水分循环分析 ----------------------------
DiffPlotVarList = [
#     # {"VAR_NAME": "RAINNC", "UNIT": "mm/day", "plotMAX1": 50, "plotMAX2": 10, "plotMIN": 0, "SCALE_FACTOR": 24, "CMAPmean":cmaps.WhiteBlueGreenYellowRed,"CMAPdiff":cmaps.amwg256, "Title" :"Precipitation rate "},    
    {"VAR_NAME": "RUNSF", "UNIT": "mm/day", "plotMAX1": 30, "plotMAX2": 20, "plotMIN": 10.00, "SCALE_FACTOR": 1, "CMAPmean":cmaps.WhiteBlueGreenYellowRed,"CMAPdiff":cmaps.CBR_drywet, "Title" :"SUFRF "},
    {"VAR_NAME": "RUNSB", "UNIT": "mm/day", "plotMAX1": 10, "plotMAX2": 10, "plotMIN": 5.00, "SCALE_FACTOR": 24*3600, "CMAPmean":cmaps.WhiteBlueGreenYellowRed,"CMAPdiff":cmaps.CBR_drywet, "Title" :"SUBRF "},
    {"VAR_NAME": "SH2O_Layer1", "UNIT": "m³/m³", "plotMAX1": 0.45, "plotMAX2": 0.05, "plotMIN": 0.2, "SCALE_FACTOR": 1, "CMAPmean":cmaps.MPL_StepSeq,"CMAPdiff":cmaps.amwg256, "Title" :"SM at 0-10cm"},
    {"VAR_NAME": "SH2O_Layer4", "UNIT": "m³/m³", "plotMAX1": 0.45, "plotMAX2": 0.05, "plotMIN": 0.2, "SCALE_FACTOR": 1, "CMAPmean":cmaps.MPL_StepSeq,"CMAPdiff":cmaps.CBR_drywet,"Title" :"SM at 100-200 cm" },

]
# 设置子图行数和列数
NumRows = 3
NumCols = 4
PlotVarList = DiffPlotVarList
SaveName = f'SoilWaterCycle_{EventState}'
TtestVar = draw_subplot(NumRows, NumCols, results_mean, PlotVarList, SaveName)


end = time.time()
print(f"Elapse Time: {end - start}Seconds")
