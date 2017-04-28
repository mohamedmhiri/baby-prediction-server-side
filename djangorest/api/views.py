from django.shortcuts import render
import json

import sys, traceback
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
#from api.models import Data
#from api.serializers import DataSerializer
from api.services import DataGenerator

@csrf_exempt
def data_list(request):
    """
    List all code data, or create a new data.
    """
    if request.method == 'GET':
        data_set = Data.objects.all()
        serializer = DataSerializer(data_set, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        entry = JSONParser().parse(request)
        print(entry)
        res = {}
        try:
            data_generator = DataGenerator()
            res = data_generator.decisionTree(entry)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("*** print_tb:")
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            print("*** print_exception:")
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
            print("*** print_exc:")
            traceback.print_exc()
            # print("*** format_exc, first and last line:")
            # formatted_lines = traceback.format_exc().splitlines()
            # print(formatted_lines[0])
            # print(formatted_lines[-1])
            # print("*** format_exception:")
            # print(repr(traceback.format_exception(exc_type, exc_value,
            #                                       exc_traceback)))
            # print("*** extract_tb:")
            # print(repr(traceback.extract_tb(exc_traceback)))
            # print("*** format_tb:")
            # print(repr(traceback.format_tb(exc_traceback)))
            # print("*** tb_lineno:", exc_traceback.tb_lineno)
        return JsonResponse({'isSick': res.tolist()[0]}, status=200)
        # serializer = DataSerializer(data=entry)
        # if serializer.is_valid():
        #     serializer.save()
        #     return JsonResponse(serializer.data, status=201)
        # return JsonResponse(serializer.errors, status=400)


@csrf_exempt
def data_detail(request, pk):
    """
    Retrieve, update or delete a code data.
    """
    try:
        data = Data.objects.get(pk=pk)
    except Data.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = DataSerializer(data)
        return JsonResponse(serializer.data)

    elif request.method == 'PUT':
        entry = JSONParser().parse(request)
        serializer = DataSerializer(data, data=entry)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':
        data.delete()
        return HttpResponse(status=204)

@csrf_exempt
def data_generator(request):
    """
    Generate a random Json data set.
    Then Convert it to csv,
    and treat them though a decision tree algorithm
    """
    if request.method == 'GET':
        # try:
        #     data_generator = DataGenerator()
        #     data_generator.exec()
        # except Exception as inst:
        #     print("error execution DataGenerator exec method:\nError Type:==>{0}\nError Args:==>{1}\nFull Error:==>{2}"
        #           .format(type(inst), inst.args, inst))
        # try:
        #     data_generator.convert()
        # except Exception as inst:
        #     print("error execution DataGenerator convert method:\nError Type:==>{0}\nError Args:==>{1}\nFull Error:==>{2}"
        #           .format(type(inst), inst.args, inst))
        try:
            data_generator = DataGenerator()
            res = data_generator.decisionTree()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("*** print_tb:")
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            print("*** print_exception:")
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
            print("*** print_exc:")
            traceback.print_exc()
            # print("*** format_exc, first and last line:")
            # formatted_lines = traceback.format_exc().splitlines()
            # print(formatted_lines[0])
            # print(formatted_lines[-1])
            # print("*** format_exception:")
            # print(repr(traceback.format_exception(exc_type, exc_value,
            #                                       exc_traceback)))
            # print("*** extract_tb:")
            # print(repr(traceback.extract_tb(exc_traceback)))
            # print("*** format_tb:")
            # print(repr(traceback.format_tb(exc_traceback)))
            # print("*** tb_lineno:", exc_traceback.tb_lineno)
        return JsonResponse({'isSick': res.tolist()[0]}, status=200)
    else:
        return JsonResponse({}, status=400)
@csrf_exempt
def random_forest(request):
    """
    Generate a random Json data set.
    Then Convert it to csv,
    and treat them though a decision tree algorithm
    """
    if request.method == 'GET':
        # try:
        #     data_generator = DataGenerator()
        #     data_generator.exec()
        # except Exception as inst:
        #     print("error execution DataGenerator exec method:\nError Type:==>{0}\nError Args:==>{1}\nFull Error:==>{2}"
        #           .format(type(inst), inst.args, inst))
        # try:
        #     data_generator.convert()
        # except Exception as inst:
        #     print("error execution DataGenerator convert method:\nError Type:==>{0}\nError Args:==>{1}\nFull Error:==>{2}"
        #           .format(type(inst), inst.args, inst))
        try:
            data_generator = DataGenerator()
            response = data_generator.exec()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("*** print_tb:")
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            print("*** print_exception:")
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
            print("*** print_exc:")
            traceback.print_exc()
            # print("*** format_exc, first and last line:")
            # formatted_lines = traceback.format_exc().splitlines()
            # print(formatted_lines[0])
            # print(formatted_lines[-1])
            # print("*** format_exception:")
            # print(repr(traceback.format_exception(exc_type, exc_value,
            #                                       exc_traceback)))
            # print("*** extract_tb:")
            # print(repr(traceback.extract_tb(exc_traceback)))
            # print("*** format_tb:")
            # print(repr(traceback.format_tb(exc_traceback)))
            # print("*** tb_lineno:", exc_traceback.tb_lineno)
        return JsonResponse(response, status=200)
    else:
        return JsonResponse({}, status=400)



