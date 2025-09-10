# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:40:55 2024

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
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.io.shapereader import Reader
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import netCDF4 as nc
from netCDF4 import Dataset

import shapefile
import time

# =============================================================================

start = time.time()
def add_Lat_and_Lon_Grid(ax, Domain, xy_labelsize):
    # 移除网格线
    ax.gridlines(draw_labels=False, linewidth=0, alpha=0)
    
    # 设置经纬度刻度
    LATmin, Latmax, LONmin, LONmax = Domain
    lon_range = int(LONmax)-int(LONmin)
    lat_range = int(Latmax)-int(LATmin)
    # print(LATmin, Latmax, LONmin, LONmax)
    # 动态计算刻度间隔
    # lon_interval = max(1, int(lon_range / 2.5))  # 假设最多12个刻度线
    # lat_interval = max(1, int(lat_range / 2.5))  # 假设最多8个刻度线
    lon_interval = 4
    lat_interval = 3
    ax.set_xticks(np.arange(int(LONmin), int(LONmax) + 1, lon_interval), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(int(LATmin), int(Latmax) + 1, lat_interval), crs=ccrs.PlateCarree())

    # 设置刻度标签格式
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    # 设置刻度线样式
    ax.tick_params(axis='both', which='major', direction='out', length=1.5, width=0.5, color='black', 
                   labelsize=xy_labelsize-0.5, labelcolor='black', pad=2.5)
    
    # 只在底部和左侧显示刻度标签
    ax.xaxis.set_tick_params(labeltop=False, labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True, labelright=False)

    # 设置轴线颜色为黑色
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    pass

def add_cbar(Cbar, ax, fig, contour, cbar_Legend_size, cbarTitle ,numticks=5):
    if Cbar==True:
        
        # # ----------------------------绘制colorbar----------------------------
        # 指定colorbar的位置和大小
        cax = fig.add_axes([0.35, 0.2, 0.3, 0.01])
        # 第一个参数：子图左下角的x坐标，表示子图距离整个图形左边界的距离占整个图形宽度的比例。
        # 第二个参数：子图左下角的y坐标，表示子图距福整个图形底边界的距离占整个图形高度的比例。
        # 第三个参数：子图的宽度，表示子图宽度占整个图形宽度的比例。
        # 第四个参数：子图的高度，表示子图高度占整个图形高度的比例。
        cbar = fig.colorbar(contour, shrink=0.6,fraction=0.04, pad=0, orientation='horizontal', ax=ax,cax=cax)#, ticks=levels)
        # cbar.ax.set_aspect(10)
    
        # cbar.set_ticks([plotMIN,(plotMAX-plotMIN)/5, (plotMAX-plotMIN)/5*2, (plotMAX-plotMIN)/5*3, (plotMAX-plotMIN)/5*4, plotMAX])
        # 设置colorbar的刻度数量为5

        
        # cbar = fig.colorbar(contour, shrink=0.8, pad=0.05)
        # cbar.ax.tick_params(labelsize = cbar_Legend_size)
        # cbar.set_ticks([-plotMAX, 0, plotMAX])
        # 设置colorbar的标题
        bbox = cbar.ax.get_position()
        title_pos = (bbox.x0 + bbox.width / 1+0.04, bbox.y0 + bbox.height / 2)
        ax.text(title_pos[0], title_pos[1], cbarTitle, fontsize=cbar_Legend_size,
                rotation=0, va='center', ha='center', transform=fig.transFigure)
        
        # Q = ax.quiver(to_np(LON), to_np(LAT), 
        #               to_np(U), to_np(V), 
        #               pivot='middle', 
        #               transform=ccrs.PlateCarree(), 
        #               regrid_shape=regrid_shape,
        #               headwidth = 9,
        #               headlength=12,
        #               scale=200) #箭头长度
        # 绘制箭头图例
        # qk = ax.quiverkey(Q, 
        #                   0.7, 0.12,                  
        #                   uvMAX, f'{uvMAX} {quiverTitle}', 
        #                   labelpos='S',
        #                   coordinates='axes',
        #                   labelsep = 0.05,
        #                   fontproperties={'size':6})
        # ax.text(title_pos[0], title_pos[1], cbarTitle, fontsize=cbar_Legend_size,
        #         rotation=0, va='center', ha='center', transform=fig.transFigure)
    elif Cbar == "subcbar":
        # fraction控制颜色条在垂直于其方向上的大小。例如，对于水平方向的颜色条，fraction 控制的是颜色条的高度；对于垂直方向的颜色条，fraction 控制的是颜色条的宽度
        cbar = fig.colorbar(contour,  ax=ax, orientation='horizontal', shrink=1.5 ,fraction=0.025, pad=0.08, aspect=35)    
                # 设置colorbar的标题
        bbox = cbar.ax.get_position()
        title_pos = (0.75,0.11)
        ax.text(title_pos[0], title_pos[1], cbarTitle, fontsize=cbar_Legend_size,
                rotation=0, va='center', ha='center', transform=fig.transFigure)
    cbar.locator = ticker.LinearLocator(numticks=numticks)
    # 设置刻度线的长度和方向
    cbar.ax.tick_params(which='both', length=1.2) #, direction='in'
    cbar.ax.tick_params(labelsize=cbar_Legend_size-0.5)
    
    
    # 自定义格式化函数
    def format_tick(tick_value, pos):
        if isinstance(tick_value, float) and not tick_value.is_integer():
            return f"{tick_value:.2f}"
        else:
            return str(int(tick_value)) if isinstance(tick_value, float) and tick_value.is_integer() else str(tick_value)
    # 自定义格式化函数
    def format_tick(tick_value, pos):
        if isinstance(tick_value, (float)):
            if tick_value > 100:
                return str(int(tick_value))
            else:
                return f"{tick_value:.2f}"

        else:
            return str(tick_value)
    
    # 使用自定义格式化函数
    cbar.formatter = ticker.FuncFormatter(format_tick)
    cbar.update_ticks()
    pass
    
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
def plot_river(ax,  river_color, max_river):
    proj = ccrs.PlateCarree()  # 创建投影
    river_floder = "/home/yfdong/data/map/river_shp"

    river_path_list = [f"{river_floder}/hyd1/hyd1.shp", 
                       f"{river_floder}/hyd3/hyd3.shp", 
                       f"{river_floder}/hyd4/hyd4.shp", 
                       f"{river_floder}/hyd5/hyd5.shp"]

    params = [
        [(0.6, 0.4)],
        [(0.5, 0.3)],
        [(0.4, 0.2)],
        [(0.3, 0.1)]
    ]


    for river_path, param in zip(river_path_list[:max_river-1], params[:max_river-1] ):
        river = shpreader.Reader(river_path).geometries()
        alpha, linewidth = param[0]
        ax.add_geometries(river, proj,facecolor='none', edgecolor=river_color, zorder=1,alpha = alpha ,linewidth = linewidth, linestyle='-')
    pass
from wrf import CoordPair
def get_CoordPair(axisName):
    if axisName == "crosswind":
        cross_start = CoordPair(lat=27, lon=122.)
        cross_end = CoordPair(lat=31.5, lon=117.)
    elif axisName == "downwind":
        cross_start = CoordPair(lat=27, lon=115.)
        cross_end = CoordPair(lat=32, lon=121.)
    else:
        print("Invalid axis name. Please use 'crosswind' or 'downwind'.")
    return cross_start, cross_end


def add_DEM_contour(ax, LON, LAT, DEM):
    width=0.9
    pdem = ax.contour(LON, LAT, DEM, levels = [300] ,alpha = width , colors = ["black"], linewidths=width)#, linestyles="dashed")
    pass
from matplotlib.colors import BoundaryNorm, ListedColormap
import cmaps
def draw_DEM(
                    fig_path ,ShpPath ,BasinShpPath,
                    Domain,
                    LON, LAT, VAR , title, save_name,
                 Height = 2.3, width=3.2,
                 dpi= 350,
                 title_labelsize = 10,xy_labelsize = 8, cbar_Legend_size = 7,
                 Grid = True, Save=False, Shp=True, River=True, Basin = True ,Lake= True):   
        # --------------------设置图片参数--------------------------
        
        # 1. 创建投影和画布
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(width, Height), dpi=dpi)
        ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
        ax.set_facecolor('#F5F5F5')

        # 2. 准备数据和水体颜色
        water_color = np.array([173/255, 216/255, 230/255, 1])  # lightblue

        # 3. 创建自定义色板（在原始色板前添加水体颜色）
        cmap = cmaps.OceanLakeLandSnow
        cmap_colors = cmap(np.linspace(0, 1, cmap.N))
        cmap_colors = np.vstack((water_color, cmap_colors))
        extended_cmap = ListedColormap(cmap_colors)

        # 4. 设置色阶和规范
        levels = np.linspace(0, 2000, 31)
        extended_levels = np.insert(levels, 0, -1)  # 添加低于0的等级
        norm = BoundaryNorm(extended_levels, ncolors=extended_cmap.N)

        # 5. 关键修改：去除掩码，直接使用原始数据
        # 将-9999值视为需要特殊处理的正常值
        contour = ax.contourf(LON, LAT, VAR,  # 使用原始VAR而不是masked_var
                            cmap=extended_cmap, 
                            levels=extended_levels,
                            norm=norm,
                            extend='both')  # 启用向下扩展

        # 6. 绘制等高线（这里需要使用掩码避免无效点）
        masked_var = np.ma.masked_where(VAR == -9999, VAR)
        pdem = ax.contour(LON, LAT, masked_var, 
                        levels=[300], 
                        alpha=0.6, 
                        colors=["black"], 
                        linewidths=0.4)

        # 7. 确保under值使用水体颜色
        contour.cmap.set_under('lightblue')

        # add_cbar("subcbar", ax, fig, contour, cbar_Legend_size, '(m)' ,numticks=5)
        cbar = fig.colorbar(contour,  ax=ax, 
                            orientation='horizontal', 
                            shrink=1.5 ,fraction=0.025, pad=0.08, aspect=35,
                            # drawedges=True,
                            # edgecolor='black',  # 设置分割线颜色
                            ticks=[0, 500, 1000, 1500, 2000]  # 手动设置刻度线
                            )    
        bbox = cbar.ax.get_position()
        title_pos = (0.78,0.115)
        ax.text(title_pos[0], title_pos[1], '(m)', fontsize=cbar_Legend_size,
                rotation=0, va='center', ha='center', transform=fig.transFigure)
        # 设置刻度线的长度和方向
        cbar.set_ticks([0, 500, 1000, 1500, 2000])  # 强制只显示这些刻度
        cbar.ax.tick_params(which='major', length=1.5)  # 设置刻度线长度
        cbar.ax.tick_params(which='minor', length=0)  # Hide minor ticks
        # cbar.ax.xaxis.set_ticks_position('none')  # Hide all other ticks
        # cbar.ax.tick_params(which='both', length=1.0) #, direction='in'
        cbar.ax.tick_params(labelsize=cbar_Legend_size-0.5)
        # # -------------------------------添加水文站点-------------------------------
        SM_ID_scatter_size = 8 #土壤湿度自动站散点大小
        H_scatter_size = 20 #水文站散点大小
        stations = {

      "Huangshan": (29.7172627, 118.3324811)  ,
    #   'HengYang': (27.3, 112.5),
    #   "ChangSha": (28.2, 112.9),
      "Ji'an": (27.3, 115.3) ,
      # "Nannanjing": (32.0615513, 118.7915619) ,
    #   "ShangRao": (28.4582821, 117.9381305)
      
    # "Xixian": (114.73309, 32.32521),
    # "Wangjiaba": (115.62635, 32.40963),
    # "Lutaizi": (116.55776, 32.54185),
    # "Fuyang": (115.85383, 32.93362),

        }
        names = list(stations.keys())  # 获取站点名称列表

        # # 提取站点的经纬度坐标
        IDlon = [coord[1] for coord in stations.values()]
        IDlat = [coord[0] for coord in stations.values()]
        # if IF_ID == True:
        # 绘制水文站点散点图
        # ax.scatter(IDlon, IDlat,s= 50, facecolor='red', edgecolors = "black", marker = "*", linewidths=1, label= "Hydrological gauge", zorder=10)
        # ax.scatter(IDlon, IDlat,s= 15, facecolor='red', edgecolors = "black", marker = "o", linewidths=1, label= "Hydrological gauge", zorder=10)

        # # 添加水文站点标签
        # for x, y, name in zip(IDlon, IDlat, names):
        #     ax.text(x + 0.1, y + 0.2, name, fontsize=5.5, ha='center', va='bottom', transform=proj)
        # =============================================================================
        GeoNames = {

      "Huaihe Basin": (33., 116)  ,
      "Yangtze Plain": (29.8, 114.7 ),
      "Dabie\nMountain": (31, 115),
    #   'Poyang\nLake Plain': (28, 116),
      "Wushan\nMountains": (31, 110),
      "Dongting\nLake Plain": (28,112.5),

        }
        names = list(GeoNames.keys())  # 获取站点名称列表

        # # 提取站点的经纬度坐标
        IDlon = [coord[1] for coord in GeoNames.values()]
        IDlat = [coord[0] for coord in GeoNames.values()]
        
        from matplotlib.font_manager import FontProperties
        # 定义所需的字体属性
        font = FontProperties(family='serif', size=6 ,weight='bold',style='italic')
        # 添加标签
        for x, y, name in zip(IDlon, IDlat, GeoNames):
            ax.text(x + 0.1, y + 0.2, name, 
                    fontproperties=font, 
                    ha='center', 
                    va='bottom', 
                    transform=proj, 
                    color='#00008B',
                    # color='#FFFF00',
                    # bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=1),
                    )
        # -------------------------------!!!绘制经纬度网格!!!--------------------------------
        if Grid == True:
            add_Lat_and_Lon_Grid(ax, Domain, xy_labelsize)
        # # ---------------------------添加shp文件----------------------------
        # # 添加行政区划
        # if Shp == True:
        #     cou=shpreader.Reader(ShpPath).geometries()
        #     ax.add_geometries(cou, proj,facecolor='none',edgecolor='black', zorder=1,alpha = 0.4,linewidth = 0.5, linestyle='--')
        # # dem=shpreader.Reader("D:/data/download_data/CN_Mountain_shp/CN_MountainDistribution/CN_MountainDistribution.shp").geometries()
        # 添加海岸线
        ax.coastlines( linewidth=0.4, color='black', alpha=0.6, linestyle='-') # 可根据需要添加或移除
        # 添加河流
        if River == True:
            river_color = "#104E8B"
            max_river = 3
            plot_river(ax,  river_color, max_river)
        # # 添加流域边界
        # if Basin == True:
        #     basin =shpreader.Reader(BasinShpPath).geometries()
        #     ax.add_geometries(basin, proj,facecolor='none',edgecolor='red', zorder=1,alpha = 0.5,linewidth = 0.3, linestyle='--')
        # # # ---------------------- 定义起点和终点 -------------------
        # cross_start, cross_end = get_CoordPair('downwind')
        # # 绘制线
        # ax.plot([cross_start.lon, cross_end.lon], [cross_start.lat, cross_end.lat], color='#104E8B', linestyle='-', linewidth=2, zorder=1)
        
        # # crosswind axis
        # cross_start, cross_end = get_CoordPair('crosswind')
        # ax.plot([cross_start.lon, cross_end.lon], [cross_start.lat, cross_end.lat], color='#CD0000', linestyle='-', linewidth=2, zorder=1)       
            
        # # -----------------------------Total--------------------------------------
        SubsetDomain = [25, 35, 109, 122]
        add_SubsetDomain(ax, SubsetDomain, linewidth=1., edgecolor='#fa503b',linestyle='--')
        font = FontProperties(family='serif', size=7 ,weight='bold')
        ax.text(SubsetDomain[2]+0.5,  # 经度位置
                SubsetDomain[1] + 0.5,  # 纬度位置（矩形框上方的偏移量）
                'Eastern China Region',  # 标注文字
                color='#CD3700',  # 文字颜色
                fontsize=7,  # 文字大小
                ha='left',  # 水平对齐方式（left：左对齐）
                va='bottom',# 垂直对齐方式（bottom：底部对齐）
                fontweight='bold',  # 字体加粗
                fontproperties=font, 
                # fontstyle='italic')  # 字体斜体)  
        )
        # SubsetDomain = [28, 34, 110, 121.5] 
        # add_SubsetDomain(ax, SubsetDomain, linewidth=1., edgecolor='red',linestyle='--')
        # ax.text(SubsetDomain[2]+0.5,  # 经度位置
        #         SubsetDomain[1] - 0.7,  # 纬度位置（矩形框上方的偏移量）
        #         'Yangtze-Huai\nRegion',  # 标注文字
        #         color='red',  # 文字颜色
        #         fontsize=7,  # 文字大小
        #         ha='left',  # 水平对齐方式（left：左对齐）
        #         va='bottom',# 垂直对齐方式（bottom：底部对齐）
        #         # fontweight='bold',  # 字体加粗
        #         # fontstyle='italic')  # 字体斜体)  
        # )
        IDlat, IDlon = (29.8, 118.5) #黄山
        SubsetDomain = [IDlat-1, IDlat+1, IDlon- 3*0.55, IDlon+ 3*0.5] #JH
        add_SubsetDomain(ax, SubsetDomain, linewidth=1., edgecolor='black',linestyle='--')
        ax.text(
                SubsetDomain[2]-0.1,  # 经度位置
                SubsetDomain[1] + 0.15,  # 纬度位置（矩形框上方的偏移量）
                'Southern Anhui\nMountainous Region',  # 标注文字
                color='#1C1C1C',  # 文字颜色
                fontsize=7,  # 文字大小
                ha='left',  # 水平对齐方式（left：左对齐）
                va='bottom',# 垂直对齐方式（bottom：底部对齐）
                fontweight='bold',  # 字体加粗
                fontproperties=font, 
                # fontstyle='italic')  # 字体斜体)  
        )
        
        SubsetDomain = [26., 28., 114, 117]
        SubsetDomain = [26.5, 28.5, 114.5, 117.5]
        font = FontProperties(family='serif', size=7 ,weight='bold')
        add_SubsetDomain(ax, SubsetDomain, linewidth=1., edgecolor='black',linestyle='--')
        ax.text(SubsetDomain[2]-1,  # 经度位置
                SubsetDomain[0] - 0.75,  # 纬度位置（矩形框上方的偏移量）
                'Jitai Basin Region',  # 标注文字
                color='#1C1C1C',  # 文字颜色
                fontsize=7,  # 文字大小
                ha='left',  # 水平对齐方式（left：左对齐）
                va='bottom',# 垂直对齐方式（bottom：底部对齐）
                fontweight='bold',  # 字体加粗
                fontproperties=font, 
                # fontstyle='italic')  # 字体斜体)  
        )
        # =============================================================================
        #-----------------------------添加指北针-------------------------------
        # add_north(ax)
        # 绘制标题
        # plt.title(f"(a) {title}" , fontsize= title_labelsize,loc="left")
        #  设置绘图范围
        LATmin, Latmax, LONmin, LONmax = Domain
        ax.set_ylim(LATmin, Latmax)
        ax.set_xlim(LONmin, LONmax)
        # 保存文件   
        # if Save == True:
        #     self._save_figure(fig_path, save_name, dpi)

        plt.show()
        plt.close() 
# ----------------------------设置共用参数----------------------------
common_params = {
    'width': 3.5*1.5,
    'Height': 2.5*1.5,
    'dpi': 1000,
    'title_labelsize': 6,
    'xy_labelsize': 6,
    'cbar_Legend_size': 6,
    'Grid': True,
    'Save': False,
    'Shp': True,
    # 'River': False,
    'River': True,
    # 'River': True,
    # 'Basin': False,
    'Basin': False,
    'Lake':True,
}
# ----------------------------设置路径----------------------------
BasinShpPath ="/home/yfdong/data/map/Huaihe_shp/output/boundary.shp"
ShpPath = "/home/yfdong/data/map/gadm41_CHN_shp/province/gadm41_CHN_1.shp"
fig_path = "/home/yfdong/data/work/LF-SAM/code/AnalysisModule/StaticDataAnalysis/Fig"
GeoPath ="/home/yfdong/data/work/LF-SAM/output/Domain/def/geo_em.nc"
nc_geo = nc.Dataset(GeoPath)
LU_INDEX = (nc_geo.variables['LU_INDEX'][0,:,:])  
SCB_DOM = (nc_geo.variables['SCB_DOM'][0,:,:])  
LANDMASK = nc_geo.variables['LANDMASK'][0,:,:]
# LANDMASK[LANDMASK==0]=-9999
DEM = nc_geo.variables['HGT_M'][0,:,:] 
DEM[LANDMASK==0]=(-9999)
LAT = (nc_geo.variables['XLAT_M'][0,:,:])  
LON = (nc_geo.variables['XLONG_M'][0,:,:]) 
Domain = [23.5,38.5,105.2,125.5]
# SubsetDomain = [27.5, 36, 111, 123]
# Domain = [27.5,32,115.5,121] 
# Domain = [27,33,114.5,122.5]
# Domain = [28.5,31,116.5,120.2]
# Domain = [28, 33, 110, 122]
#------------------------------绘图--------------------------------
# import sys
# # 将目标模块所在的路径添加到 sys.path 中
# sys.path.append(r"/home/yfdong/data/work/LF-SAM/code/Library")
# from MeteoVarPlot import draw_2DVAR, draw_Bias, draw_VARtemporal, draw_Bias_subplot,draw_2DVAR_subplot ,draw_2DuvVAR ,draw_uvBias_subplot ,draw_2DuvVAR_subplot
# from MeteoChartPlot import add_TickGrid
# from PreprocessVar import (
#     get_MONTH_abbr, get_pentad, get_letter,
#     preprocess_var, read_var,Events_configs,Meiyu_configs,
#     Read_VarsFromVarsDict, experiments_types,return_Event_Date,
#     load_geodata, SubsetDomainConfigure ,create_domain_mask,get_letter,
#     get_CoordPair, JJA_configs
# )
# proj = ccrs.PlateCarree() 
# VAR = DEM*LANDMASK 
# # proj = ccrs.LambertConformal()  # 创建兰波托投影
# fig = plt.figure(figsize=(6, 8), dpi=300)
# ax = fig.subplots(1, 1, subplot_kw={'projection':proj})
# import cmaps
# cmap = cmaps.OceanLakeLandSnow
# # 屏蔽 VAR < -999 的部分
# masked_VAR = np.ma.masked_where(VAR < -999, VAR)
# levels = np.linspace(0, 2000, 31)
# # 绘制主数据
# # contour = ax.contourf(
# #     LON[:, :], LAT[:, :], masked_VAR, 
# #     cmap=cmap, 
# #     levels=levels, 
# #     extend='max',
# # )

# # 单独绘制 VAR < -999 的区域为浅蓝色
# ax.contourf(
#     LON[:, :], LAT[:, :], VAR, 
#     # colors='blue',  # 直接指定颜色
#     # levels=[0.5, 1.5],   # 确保只填充 True (1) 的区域
#     # hatches=['...'],     # 可选：添加斜线等标记
#     alpha=0.5,           # 可选：调整透明度
# )

# # LANDMASK[LANDMASK==0]=np.nan
# VAR = LANDMASK*DEM
# # 创建掩码数组，将 VAR==-9999 的值设为 NaN
# # masked_var = np.ma.masked_where(VAR <= -9999, VAR)
# plt.imshow(DEM)
draw_DEM(
                fig_path ,ShpPath ,BasinShpPath,
                Domain,
                LON, LAT, DEM, title = "DEM", save_name = 'DEM.png',
                **common_params
)
# viz.draw_USGS_LUINDEX(
#                 fig_path ,ShpPath ,BasinShpPath,
#                 Domain,
#                 LON, LAT, LU_INDEX , title = 'LUINDEX', save_name = 'LUINDEX.png',
#                 **common_params
#                 )


# viz.draw_SCT(
#                 fig_path ,ShpPath ,BasinShpPath,
#                 Domain,
#                 LON, LAT, SCB_DOM , title = 'SCT', save_name = 'SCT.png',
#                 **common_params
# )










end = time.time()
print(f"Elapse Time: {end - start}Seconds")
